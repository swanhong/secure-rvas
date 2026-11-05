package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	securecrypto "github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func PackBeta(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	beta mpc_core.RVec,
) securecrypto.CipherVector {
	coefficientCount := len(beta)
	slots := heParams.GetSlots()
	packed := mpc_core.InitRMat(mpcObj.GetRType().Zero(), coefficientCount, slots)
	for coefficient := range beta {
		for slot := 0; slot < slots; slot++ {
			packed[coefficient][slot] = beta[coefficient].Copy()
		}
	}

	packedMatrix := mpcObj.SSToCMat(heParams, packed)
	packedBeta := make(securecrypto.CipherVector, coefficientCount)
	if mpcObj.GetPid() == auxiliaryPartyID {
		return packedBeta
	}
	for coefficient := range packedBeta {
		packedBeta[coefficient] = packedMatrix[coefficient][0]
	}
	return packedBeta
}

func ComputeGtX(
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	gp []*mat.Dense,
	x *mat.Dense,
	observe func(string) func(),
) ([]*mat.Dense, securecrypto.PlainVector) {
	done := observe("local_gtx")
	gtx := make([]*mat.Dense, len(batch.GeneIndices))
	for position, geneIndex := range batch.GeneIndices {
		gtx[position] = new(mat.Dense)
		gtx[position].Mul(gp[geneIndex].T(), x)
	}
	done()

	_, covariateCount := x.Dims()
	gtxEncoded := make(securecrypto.PlainVector, covariateCount)
	for covariate := 0; covariate < covariateCount; covariate++ {
		columnByGene := make([][]float64, len(batch.GeneIndices))
		for position := range batch.GeneIndices {
			columnByGene[position] = mat.Col(nil, covariate, gtx[position])
		}
		packed := PackGeneBatch(dataParams, cryptoParams, batch, columnByGene)
		encoded, _ := securecrypto.EncodeFloatVector(heParams, packed)
		gtxEncoded[covariate] = encoded[0]
	}
	return gtx, gtxEncoded
}

func ComputeScore(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	gp []*mat.Dense,
	y0 mat.Vector,
	gtxEncoded securecrypto.PlainVector,
	packedBeta securecrypto.CipherVector,
	embeddings [][]mpc_core.RMat,
	observe func(string) func(),
) securecrypto.CipherVector {
	/*
		localScore[k] = Enc(gty[k,t]) - Σa Enc(beta[a,t]) * Plain(gtx[k,a])
		score = Σk E_kᵀ localScore[k], in the common coordinates
	*/
	if mpcObj.GetPid() == auxiliaryPartyID {
		return alignScore(mpcObj, heParams, dataParams, cryptoParams, batch, nil, embeddings)
	}

	done := observe("local_gty")
	gty := make([][]float64, len(batch.GeneIndices))
	for position, geneIndex := range batch.GeneIndices {
		capM := dataParams.Genes[geneIndex].Cap
		gty[position] = make([]float64, capM)
		for variant := 0; variant < capM; variant++ {
			gty[position][variant] = mat.Dot(gp[geneIndex].ColView(variant), y0)
		}
	}
	done()

	packedGty := PackGeneBatch(dataParams, cryptoParams, batch, gty)
	localScore, _ := securecrypto.EncryptFloatVector(heParams, packedGty)
	for covariate := range packedBeta {
		term := securecrypto.CPMult(
			heParams,
			securecrypto.CipherVector{packedBeta[covariate]},
			securecrypto.PlainVector{gtxEncoded[covariate]},
		)
		localScore = securecrypto.CSub(heParams, localScore, term)
	}
	return alignScore(mpcObj, heParams, dataParams, cryptoParams, batch, localScore, embeddings)
}

// alignScore converts every site's local score to shares, aligns and sums them,
// then encrypts the combined score in the same batch packing.
func alignScore(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	localScore securecrypto.CipherVector,
	embeddings [][]mpc_core.RMat,
) securecrypto.CipherVector {
	rtype := mpcObj.GetRType()
	siteScores := make([][]mpc_core.RMat, mpcObj.GetNParty()-1)
	for source := 1; source < mpcObj.GetNParty(); source++ {
		packed := mpcObj.CVecToSS(heParams, rtype, localScore, source, 1, cryptoParams.Slots)
		for _, values := range UnpackGeneBatch(dataParams, cryptoParams, batch, packed) {
			column := make(mpc_core.RMat, len(values))
			for variant, value := range values {
				column[variant] = mpc_core.RVec{value}
			}
			siteScores[source-1] = append(siteScores[source-1], column)
		}
	}

	aligned := alignShared(
		mpcObj, batchEmbeddings(batch, embeddings), siteScores,
	)

	values := make([][]mpc_core.RElem, len(aligned))
	for position, column := range aligned {
		for _, row := range column {
			values[position] = append(values[position], row[0])
		}
	}
	return mpcObj.SSToCVec(heParams, packSharedGeneBatch(rtype, dataParams, cryptoParams, batch, values))
}

func PackWeight(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	signedWeight []mpc_core.RVec,
) securecrypto.CipherVector {
	weightByGene := make([][]mpc_core.RElem, len(batch.GeneIndices))
	for position, geneIndex := range batch.GeneIndices {
		weightByGene[position] = signedWeight[geneIndex]
	}
	packed := packSharedGeneBatch(
		mpcObj.GetRType(),
		dataParams,
		cryptoParams,
		batch,
		weightByGene,
	)
	return mpcObj.SSToCVec(heParams, packed)
}

func ReduceQL(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	score securecrypto.CipherVector,
	packedWeight securecrypto.CipherVector,
) (geneBatchQ, geneBatchL mpc_core.RVec) {
	/*
		For each gene g and current phenotype t,
		with variants indexed by j:

			u[g,j,t] = signedWeight[g,j] * score[g,j,t] / sqrt(N)
			L[g,t]   = Σj u[g,j,t]
			Q[g,t]   = Σj u[g,j,t]^2

		The inner sum at gene g's lane adds only that gene's slots, so other slots need no mask.
	*/
	// u = weighted score
	var encPackedU securecrypto.CipherVector
	if mpcObj.GetPid() != auxiliaryPartyID {
		encPackedU = securecrypto.CMult(heParams, score, packedWeight)
	}

	packedUShares := mpcObj.CVecToSS(
		heParams,
		mpcObj.GetRType(),
		encPackedU,
		auxiliaryPartyID,
		1,
		cryptoParams.Slots,
	)
	uSharesByGene := UnpackGeneBatch(
		dataParams, cryptoParams, batch, packedUShares,
	)

	// L = sum of u shares by variants
	geneBatchL = mpc_core.InitRVec(
		mpcObj.GetRType().Zero(), len(batch.GeneIndices),
	)
	for position := range uSharesByGene {
		for _, value := range uSharesByGene[position] {
			geneBatchL[position] = geneBatchL[position].Add(value)
		}
	}

	repackedUshares := packSharedGeneBatch(
		mpcObj.GetRType(),
		dataParams,
		cryptoParams,
		batch,
		uSharesByGene,
	)
	freshEncPackedU := mpcObj.SSToCVec(
		heParams, repackedUshares,
	)

	var encQRows securecrypto.CipherVector
	if mpcObj.GetPid() != auxiliaryPartyID {
		encUSquare := securecrypto.CMult(
			heParams, freshEncPackedU, freshEncPackedU,
		)

		encQRows = make(securecrypto.CipherVector, 1)
		encQRows[0] = encUSquare[0].CopyNew()
		nu := cryptoParams.Slots / batch.W
		if err := heParams.WithEvaluator(func(evaluator *ckks.Evaluator) error {
			return evaluator.InnerSum(
				encUSquare[0], nu, batch.W, encQRows[0],
			)
		}); err != nil {
			panic(err)
		}
	}

	nu := cryptoParams.Slots / batch.W
	qRowShares := mpcObj.CVecToSS(
		heParams,
		mpcObj.GetRType(),
		encQRows,
		auxiliaryPartyID,
		1,
		nu,
	)
	h := nu / len(batch.GeneIndices)
	geneBatchQ = mpc_core.InitRVec(
		mpcObj.GetRType().Zero(), len(batch.GeneIndices),
	)
	for position := range geneBatchQ {
		geneBatchQ[position] = qRowShares[position*h].Copy()
	}
	return geneBatchQ, geneBatchL
}

func ComputeGpQL(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	gp []*mat.Dense,
	y0 mat.Vector,
	gtxEncoded securecrypto.PlainVector,
	packedBeta securecrypto.CipherVector,
	packedWeight securecrypto.CipherVector,
	embeddings [][]mpc_core.RMat,
	observe func(string) func(),
) (geneBatchQ, geneBatchL mpc_core.RVec) {
	score := ComputeScore(
		mpcObj,
		heParams,
		dataParams,
		cryptoParams,
		batch,
		gp,
		y0,
		gtxEncoded,
		packedBeta,
		embeddings,
		observe,
	)
	return ReduceQL(
		mpcObj,
		heParams,
		dataParams,
		cryptoParams,
		batch,
		score,
		packedWeight,
	)
}
