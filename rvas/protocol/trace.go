package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

// newPublicGtGAction returns R ↦ Σk E_kᵀ D̄_k (E_k R), where D̄_k is
// site k's local GᵀG/n and E_1 is identity. E products need no truncation;
// each fixed-point D̄_k product is truncated once. No pooled Gram is formed.
func newPublicGtGAction(
	mpcObj *mpc.MPC, dataParams DataParams, batch GeneBatch,
	localGtG []*mat.Dense, embeddings [][]mpc_core.RMat,
) func([]mpc_core.RMat) []mpc_core.RMat {
	shapes := make([][2]int, len(batch.GeneIndices))
	for position, geneIndex := range batch.GeneIndices {
		capM := dataParams.Genes[geneIndex].Cap
		shapes[position] = [2]int{capM, capM}
	}
	sites := shareSites(mpcObj, localGtG, shapes)
	selected := batchEmbeddings(batch, embeddings)
	var siteGtG, nonAnchor []mpc_core.RMat
	for site := range sites {
		siteGtG = append(siteGtG, sites[site]...)
		if site > 0 {
			nonAnchor = append(nonAnchor, selected[site]...)
		}
	}
	return func(right []mpc_core.RMat) []mpc_core.RMat {
		var repeated []mpc_core.RMat
		for site := 1; site < len(sites); site++ {
			repeated = append(repeated, right...)
		}
		moved := mpcObj.SSMultMats(nonAnchor, repeated)
		products := multiplyTruncated(mpcObj, siteGtG, append(append([]mpc_core.RMat(nil), right...), moved...))
		bySite := make([][]mpc_core.RMat, len(sites))
		for site := range bySite {
			bySite[site] = products[site*len(right) : (site+1)*len(right)]
		}
		return alignShared(mpcObj, selected, bySite)
	}
}

func ComputeGeneBatchKernelStatistics(
	mpcObj *mpc.MPC,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	localGtx []*mat.Dense,
	localGtG []*mat.Dense,
	omega mpc_core.RMat,
	weight []mpc_core.RVec,
	signedWeight []mpc_core.RVec,
	embeddings [][]mpc_core.RMat,
	seed int64,
	observe func(stage string) func(),
) (geneBatchV, geneBatchInvS1, geneBatchS2, geneBatchS3 mpc_core.RVec) {
	/*
		Compute the phenotype-independent kernel statistics for one gene batch.

		Inputs in global gene:
		    weight, signedWeight

		Inputs in batch:
		    localGtx, localGtG

		For each gene, in the common coordinates:
		    pooledGtx = Σk E_kᵀ * Transpose(Gp[k]) * X[k] / N
		    pooledGtG = Σk E_kᵀ * Transpose(Gp[k]) * Gp[k] * E_k / N

		With unscaled D = GᵀG and Gtx = GᵀX:
		    Kbar  = Diag(weight) * (D - Gtx*(XᵀX)^-1*Gtxᵀ) * Diag(weight) / (2*N)
		    invS1 = 1 / Trace(Kbar)
		    S2    = Trace(Kbar²) / Trace(Kbar)²
		    S3    = Trace(Kbar³) / Trace(Kbar)³
		    V  = Burden variance / N

		S2 and S3 use probe estimates when the cap exceeds the probe budget.
		Outputs V, invS1, S2 and S3 remain secret-shared and use batch order.
	*/
	geneCount := len(batch.GeneIndices)

	// 1. Share pooledGtx in the common coordinates.
	// pooledGtx = Σk E_kᵀ * Transpose(G[k]) * X[k] / N
	done := observe("kernel_inputs")
	pooledGtx := SharePooledGtx(mpcObj, dataParams, batch, localGtx, embeddings)

	// 2. Share diagGtG in the common coordinates.
	// diagGtG = Σk E_kᵀ diag(localGtG[k]), the diagonal of pooledGtG because
	// real rows select distinct coordinates and dummy Gram rows/columns are zero.
	localDiag := make([]*mat.Dense, geneCount)
	if mpcObj.GetPid() != auxiliaryPartyID {
		for position, geneIndex := range batch.GeneIndices {
			capM := dataParams.Genes[geneIndex].Cap
			localDiag[position] = mat.NewDense(capM, 1, nil)
			for variant := 0; variant < capM; variant++ {
				localDiag[position].Set(variant, 0, localGtG[position].At(variant, variant))
			}
		}
	}

	diagGtG := make([]mpc_core.RVec, geneCount)
	for position, aligned := range AlignShares(mpcObj, batchEmbeddings(batch, embeddings), localDiag, 1) {
		for _, row := range aligned {
			diagGtG[position] = append(diagGtG[position], row[0])
		}
	}
	done()

	// 3. Compute normalized moments and reuse the first GtG action for gtgWeight.
	geneBatchInvS1, geneBatchS2, geneBatchS3, gtgWeight := ComputeKernelTraces(
		mpcObj, dataParams, cryptoParams, batch,
		localGtG, pooledGtx, diagGtG, omega, weight, signedWeight, embeddings, seed, observe,
	)

	// 4. Compute the normalized Burden variance using gtgWeight = pooledGtG * signedWeight.
	done = observe("burden_variance")
	geneBatchV = ComputeBurdenVariance(
		mpcObj, batch, pooledGtx, signedWeight, gtgWeight, omega,
	)
	done()

	return geneBatchV, geneBatchInvS1, geneBatchS2, geneBatchS3
}

func ComputeKernelTraces(
	mpcObj *mpc.MPC,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	localGtG []*mat.Dense,
	pooledGtx []mpc_core.RMat,
	diagGtG []mpc_core.RVec,
	omega mpc_core.RMat,
	weight []mpc_core.RVec,
	signedWeight []mpc_core.RVec,
	embeddings [][]mpc_core.RMat,
	seed int64,
	observe func(stage string) func(),
) (geneBatchInvS1, geneBatchS2, geneBatchS3 mpc_core.RVec, gtgWeight []mpc_core.RVec) {
	/*
		Compute the three kernel traces and the public GtG burden action.

		Inputs in global gene order:
		    weight, signedWeight

		Inputs in batch order:
		    localGtG, pooledGtx, diagGtG

		Outputs in batch order:
		    1/S1, S2/S1^2, S3/S1^3, gtgWeight
	*/
	geneCount := len(batch.GeneIndices)
	rtype := mpcObj.GetRType()
	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()

	// 1. Generate the trace probes.
	traceProbe := make([]*mat.Dense, geneCount)
	probeScale := make([]float64, geneCount)

	for position, geneIndex := range batch.GeneIndices {
		traceProbe[position], probeScale[position] =
			TraceProbe(dataParams.Genes[geneIndex].Cap, cryptoParams.R, seed)
	}

	// 2. Compute theta and weightSquared for each gene.
	//
	// theta         = pooledGtx * omega
	// weightSquared = weight .* weight
	batchWeight := make([]mpc_core.RVec, geneCount)
	batchSignedWeight := make([]mpc_core.RVec, geneCount)
	theta := make([]mpc_core.RMat, geneCount)
	weightSquared := make([]mpc_core.RVec, geneCount)

	for position, geneIndex := range batch.GeneIndices {
		batchWeight[position] = weight[geneIndex]
		batchSignedWeight[position] = signedWeight[geneIndex]

		theta[position] = mpcObj.SSMultMat(pooledGtx[position], omega)
		theta[position] = mpcObj.TruncMat(theta[position], dataBits, fracBits)

		weightSquared[position] = mpcObj.SSMultElemVec(batchWeight[position], batchWeight[position])
		weightSquared[position] = mpcObj.TruncVec(weightSquared[position], dataBits, fracBits)
	}

	// 3. Compute S1 before the GtG actions.
	flattenMatrix := func(matrix mpc_core.RMat) mpc_core.RVec {
		rows, columns := matrix.Dims()
		values := mpc_core.InitRVec(rtype.Zero(), rows*columns)

		offset := 0
		for row := 0; row < rows; row++ {
			for column := 0; column < columns; column++ {
				values[offset] = matrix[row][column].Copy()
				offset++
			}
		}
		return values
	}

	geneBatchS1 := mpc_core.InitRVec(rtype.Zero(), geneCount)
	for position := range batch.GeneIndices {
		weightedTheta := scaleSharedRows(mpcObj, weightSquared[position], theta[position])
		diagTerm := sharedDot(mpcObj, weightSquared[position], diagGtG[position])
		projectionTerm := sharedDot(
			mpcObj, flattenMatrix(weightedTheta), flattenMatrix(pooledGtx[position]),
		)

		scaled := mpc_core.RVec{
			diagTerm.Sub(projectionTerm).Mul(rtype.FromFloat64(0.5, fracBits)),
		}
		geneBatchS1[position] = mpcObj.TruncVec(scaled, dataBits, fracBits)[0]
	}

	publicOne := mpc_core.InitRVec(rtype.Zero(), geneCount)
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		publicOne.AddScalar(rtype.FromFloat64(1, fracBits))
	}
	geneBatchInvS1 = mpcObj.Divide(publicOne, geneBatchS1, false)

	scaleByGene := func(matrix mpc_core.RMat, scalar mpc_core.RElem) mpc_core.RMat {
		rowScale := mpc_core.InitRVec(rtype.Zero(), len(matrix))
		for row := range rowScale {
			rowScale[row] = scalar.Copy()
		}
		return scaleSharedRows(mpcObj, rowScale, matrix)
	}

	// 4. Build the first combined GtG right-hand side.
	// weightedProbe = Diag(weight) * traceProbe
	// firstRight = ConcatColumns(weightedProbe, signedWeight)
	firstRight := make([]mpc_core.RMat, geneCount)
	weightedProbe := make([]mpc_core.RMat, geneCount)

	for position, geneIndex := range batch.GeneIndices {
		capM := dataParams.Genes[geneIndex].Cap

		_, probeCount := traceProbe[position].Dims()

		sharedProbe := mpc_core.InitRMat(rtype.Zero(), capM, probeCount)
		if mpcObj.GetPid() == mpcObj.GetHubPid() {
			for variant := 0; variant < capM; variant++ {
				for column := 0; column < probeCount; column++ {
					sharedProbe[variant][column] =
						rtype.FromFloat64(traceProbe[position].At(variant, column), fracBits)
				}
			}
		}

		weightedProbe[position] = scaleSharedRows(mpcObj, batchWeight[position], sharedProbe)

		firstRight[position] = mpc_core.InitRMat(rtype.Zero(), capM, probeCount+1)
		for variant := 0; variant < capM; variant++ {
			for column := 0; column < probeCount; column++ {
				firstRight[position][variant][column] = weightedProbe[position][variant][column].Copy()
			}
			firstRight[position][variant][probeCount] = batchSignedWeight[position][variant].Copy()
		}
	}

	// 5. Compute the first public GtG action.
	// firstGtgAction = pooledGtG * firstRight
	done := observe("first_gtg_action")
	gtgAction := newPublicGtGAction(mpcObj, dataParams, batch, localGtG, embeddings)
	firstGtgAction := gtgAction(firstRight)
	done()

	// 6. Split the first action and compute kProbe.
	// kProbe = Kpp * traceProbe
	// gtgWeight = pooledGtG * signedWeight (pooledGtG is already divided by N)
	kProbe := make([]mpc_core.RMat, geneCount)
	gtgWeight = make([]mpc_core.RVec, geneCount)

	for position, geneIndex := range batch.GeneIndices {
		capM := dataParams.Genes[geneIndex].Cap

		_, probeCount := traceProbe[position].Dims()

		gtgProbe := mpc_core.InitRMat(rtype.Zero(), capM, probeCount)
		gtgWeight[position] = mpc_core.InitRVec(rtype.Zero(), capM)

		for variant := 0; variant < capM; variant++ {
			for column := 0; column < probeCount; column++ {
				gtgProbe[variant][column] = firstGtgAction[position][variant][column].Copy()
			}
			gtgWeight[position][variant] = firstGtgAction[position][variant][probeCount].Copy()
		}

		kProbe[position] = ComputeKppRight(
			mpcObj, weightedProbe[position], gtgProbe,
			pooledGtx[position], theta[position], batchWeight[position],
		)
		kProbe[position] = scaleByGene(
			kProbe[position], geneBatchInvS1[position],
		)
	}

	// 7. Compute the dependent second public GtG action.
	// weightedKProbe = Diag(weight) * normalized kProbe
	// secondGtgAction = pooledGtG * weightedKProbe
	weightedKProbe := make([]mpc_core.RMat, geneCount)
	for position := range batch.GeneIndices {
		weightedKProbe[position] = scaleSharedRows(mpcObj, batchWeight[position], kProbe[position])
	}

	done = observe("second_gtg_action")
	secondGtgAction := gtgAction(weightedKProbe)
	done()

	kSquaredProbe := make([]mpc_core.RMat, geneCount)
	for position := range batch.GeneIndices {
		kSquaredProbe[position] = ComputeKppRight(
			mpcObj, weightedKProbe[position], secondGtgAction[position],
			pooledGtx[position], theta[position], batchWeight[position],
		)
		kSquaredProbe[position] = scaleByGene(
			kSquaredProbe[position], geneBatchInvS1[position],
		)
	}

	// 8. Compute the normalized traces.
	// S2/S1^2 = probeScale * Dot(KZ/S1, KZ/S1)
	// S3/S1^3 = probeScale * Dot(KZ/S1, K^2Z/S1^2)
	geneBatchS2 = mpc_core.InitRVec(rtype.Zero(), geneCount)
	geneBatchS3 = mpc_core.InitRVec(rtype.Zero(), geneCount)

	for position := range batch.GeneIndices {
		flatKProbe := flattenMatrix(kProbe[position])
		tau2Term := sharedDot(mpcObj, flatKProbe, flatKProbe)
		tau3Term := sharedDot(mpcObj, flatKProbe, flattenMatrix(kSquaredProbe[position]))

		scaledTau := mpc_core.RVec{
			tau2Term.Mul(rtype.FromFloat64(probeScale[position], fracBits)),
			tau3Term.Mul(rtype.FromFloat64(probeScale[position], fracBits)),
		}
		scaledTau = mpcObj.TruncVec(scaledTau, dataBits, fracBits)

		geneBatchS2[position] = scaledTau[0]
		geneBatchS3[position] = scaledTau[1]
	}

	return geneBatchInvS1, geneBatchS2, geneBatchS3, gtgWeight
}
