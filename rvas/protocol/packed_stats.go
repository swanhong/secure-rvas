package protocol

import (
	"math"
	"sync"

	mpc_core "github.com/hhcho/mpc-core"
	securecrypto "github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

type geneBatchTerms struct {
	localGtx     []*mat.Dense
	gtxEncoded   securecrypto.PlainVector
	localGtG     []*mat.Dense
	packedWeight securecrypto.CipherVector
}

func ComputePackedStatistics(
	mpcObjects []*mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	loadGenotypes func(GeneBatch) ([]*mat.Dense, error),
	x *mat.Dense,
	y0 *mat.Dense,
	beta mpc_core.RMat,
	xtxInv mpc_core.RMat,
	weight []mpc_core.RVec,
	signedWeight []mpc_core.RVec,
	embeddings [][]mpc_core.RMat,
	seed int64,
	laneObservers []func(stage string) func(),
	observeBatch func(width, geneCount int) func(),
) (
	geneQ, geneL mpc_core.RMat,
	geneV, geneInvS1, geneS2, geneS3 mpc_core.RVec,
	err error,
) {
	/*
		Compute all packed phenotype and kernel statistics.

		For every gene g and phenotype t:
		    geneQ[g,t] = Q divided by N
		    geneL[g,t] = L divided by sqrt(N)

		For every gene g:
		    V[g]     = Burden variance / N
		    invS1[g] = 1 / Trace(K/N)
		    S2[g]    = Trace((K/N)²) / Trace(K/N)²
		    S3[g]    = Trace((K/N)³) / Trace(K/N)³

		All outputs remain secret-shared and use global gene order.
	*/
	setupMPC := mpcObjects[0]
	geneCount := len(dataParams.Genes)
	phenotypeCount := dataParams.PhenotypeCount
	rtype := setupMPC.GetRType()

	geneQ = mpc_core.InitRMat(rtype.Zero(), geneCount, phenotypeCount)
	geneL = mpc_core.InitRMat(rtype.Zero(), geneCount, phenotypeCount)

	geneV = mpc_core.InitRVec(rtype.Zero(), geneCount)
	geneInvS1 = mpc_core.InitRVec(rtype.Zero(), geneCount)
	geneS2 = mpc_core.InitRVec(rtype.Zero(), geneCount)
	geneS3 = mpc_core.InitRVec(rtype.Zero(), geneCount)

	// Omega' = N * (X^T X)^-1 keeps the normalized kernel projection balanced.
	omega := xtxInv.Copy()
	omega.MulScalar(rtype.FromInt(dataParams.N))

	// 1. Pack each shared beta column once for reuse by every gene batch.
	packedBeta := make([]securecrypto.CipherVector, phenotypeCount)

	done := laneObservers[0]("beta_packing")
	for phenotype := 0; phenotype < phenotypeCount; phenotype++ {
		betaColumn := mpc_core.InitRVec(rtype.Zero(), dataParams.C)
		for covariate := 0; covariate < dataParams.C; covariate++ {
			betaColumn[covariate] = beta[covariate][phenotype].Copy()
		}
		packedBeta[phenotype] = PackBeta(setupMPC, heParams, betaColumn)
	}
	done()

	batches := cryptoParams.Batches
	workerCount := len(mpcObjects)
	if workerCount > len(batches) {
		workerCount = len(batches)
	}

	batchErrors := make([]error, len(batches))
	var workers sync.WaitGroup
	workers.Add(workerCount)

	for lane := 0; lane < workerCount; lane++ {
		go func(lane int) {
			defer workers.Done()

			mpcObj := mpcObjects[lane]
			observe := laneObservers[lane]

			for batchIndex := lane; batchIndex < len(batches); batchIndex += workerCount {
				batch := batches[batchIndex]
				doneLoad := observe("genotype_loading")
				gp, loadErr := loadGenotypes(batch)
				doneLoad()
				if loadErr != nil {
					batchErrors[batchIndex] = loadErr
					return
				}

				doneBatch := observeBatch(batch.W, len(batch.GeneIndices))

				// 2. Prepare phenotype-independent values.
				done := observe("batch_preparation")
				terms := PrepareGeneBatch(
					mpcObj,
					heParams,
					dataParams,
					cryptoParams,
					batch,
					gp,
					x,
					signedWeight,
					observe,
				)
				done()

				// 3. Compute phenotype-dependent Q and L.
				for phenotype := 0; phenotype < phenotypeCount; phenotype++ {
					var yColumn mat.Vector
					if mpcObj.GetPid() != auxiliaryPartyID {
						yColumn = y0.ColView(phenotype)
					}

					done = observe("public_ql")
					batchGpQ, batchGpL := ComputeGpQL(
						mpcObj,
						heParams,
						dataParams,
						cryptoParams,
						batch,
						gp,
						yColumn,
						terms.gtxEncoded,
						packedBeta[phenotype],
						terms.packedWeight,
						embeddings,
						observe,
					)
					done()

					for position, geneIndex := range batch.GeneIndices {
						geneQ[geneIndex][phenotype] = batchGpQ[position].Copy()
						geneL[geneIndex][phenotype] = batchGpL[position].Copy()
					}
				}

				// 4. Compute phenotype-independent variance and traces.
				batchV, batchInvS1, batchS2, batchS3 :=
					ComputeGeneBatchKernelStatistics(
						mpcObj,
						dataParams,
						cryptoParams,
						batch,
						terms.localGtx,
						terms.localGtG,
						omega,
						weight,
						signedWeight,
						embeddings,
						seed,
						observe,
					)

				for position, geneIndex := range batch.GeneIndices {
					geneV[geneIndex] = batchV[position].Copy()
					geneInvS1[geneIndex] = batchInvS1[position].Copy()
					geneS2[geneIndex] = batchS2[position].Copy()
					geneS3[geneIndex] = batchS3[position].Copy()
				}

				doneBatch()
			}
		}(lane)
	}

	workers.Wait()

	for _, batchErr := range batchErrors {
		if batchErr != nil {
			err = batchErr
			return
		}
	}

	return
}

func PrepareGeneBatch(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	gp []*mat.Dense,
	x *mat.Dense,
	signedWeight []mpc_core.RVec,
	observe func(string) func(),
) geneBatchTerms {
	/*
		Prepare the phenotype-independent values reused by one gene batch.

		Cohort-local:
		    localGtx = Transpose(Gp) * X
		    localGtG = Transpose(Gp) * Gp / N

		Encrypted:
		    packedWeight = signedWeight / sqrt(N)
	*/
	var terms geneBatchTerms

	// 1. Compute cohort-local public-variant terms.
	if mpcObj.GetPid() != auxiliaryPartyID {
		terms.localGtx, terms.gtxEncoded = ComputeGtX(
			heParams, dataParams, cryptoParams, batch, gp, x, observe,
		)
		done := observe("local_gtg")
		terms.localGtG = ComputeLocalGtG(dataParams, batch, gp)
		done()
	}

	// 2. Pack the shared public-variant weights and scale them by 1/sqrt(N). Unused slots are
	// already zero.
	terms.packedWeight = PackWeight(
		mpcObj, heParams, dataParams, cryptoParams, batch, signedWeight,
	)
	if mpcObj.GetPid() != auxiliaryPartyID {
		terms.packedWeight = securecrypto.CMultConst(
			heParams, terms.packedWeight, 1/math.Sqrt(float64(dataParams.N)), false,
		)
	}

	return terms
}
