package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

func ComputeLocalGtG(
	dataParams DataParams,
	batch GeneBatch,
	gp []*mat.Dense,
) []*mat.Dense {
	/*
		For each gene g:
		localGtG[g] = Transpose(Gp[g]) * Gp[g] / N
	*/
	invN := 1 / float64(dataParams.N)
	localGtG := make([]*mat.Dense, len(batch.GeneIndices))
	for position, geneIndex := range batch.GeneIndices {
		localGtG[position] = new(mat.Dense)
		localGtG[position].Mul(gp[geneIndex].T(), gp[geneIndex])
		localGtG[position].Scale(invN, localGtG[position])
	}
	return localGtG
}

func SharePooledGtx(
	mpcObj *mpc.MPC,
	dataParams DataParams,
	batch GeneBatch,
	localGtx []*mat.Dense,
	embeddings [][]mpc_core.RMat,
) []mpc_core.RMat {
	/*
		For each gene g, in the common coordinates:
		pooledGtx[g]
		    = Σk E_kᵀ * Transpose(G[k,g]) * X[k] / N
		The pooled result remains secret-shared.
	*/
	scaled := make([]*mat.Dense, len(batch.GeneIndices))
	for position := range batch.GeneIndices {
		if localGtx != nil {
			scaled[position] = new(mat.Dense)
			scaled[position].Scale(1/float64(dataParams.N), localGtx[position])
		}
	}
	return AlignShares(mpcObj, batchEmbeddings(batch, embeddings), scaled, dataParams.C)
}

// multiplyTruncated returns left[i] * right[i] for fixed-point pairs with one batched multiplication
// and one truncation.
func multiplyTruncated(mpcObj *mpc.MPC, left, right []mpc_core.RMat) []mpc_core.RMat {
	products := mpcObj.SSMultMats(left, right)
	var flat mpc_core.RVec
	for _, product := range products {
		for _, row := range product {
			flat = append(flat, row...)
		}
	}
	truncated := mpcObj.TruncVec(flat, mpcObj.GetDataBits(), mpcObj.GetFracBits())
	offset := 0
	for _, product := range products {
		for row := range product {
			end := offset + len(product[row])
			product[row] = truncated[offset:end:end]
			offset = end
		}
	}
	return products
}

func ComputeBurdenQuadratic(
	mpcObj *mpc.MPC,
	signedWeight mpc_core.RVec,
	gtgWeight mpc_core.RVec,
) mpc_core.RElem {
	/*
		For one gene g, with w = signedWeight:
		gtgWeight = (Transpose(G) * G / N) * w
		quadratic = Dot(w, gtgWeight)
	*/
	return sharedDot(mpcObj, signedWeight, gtgWeight)
}

func ComputeBurdenProjectionTerm(
	mpcObj *mpc.MPC,
	pooledGtx mpc_core.RMat,
	signedWeight mpc_core.RVec,
	omega mpc_core.RMat,
) mpc_core.RElem {
	/*
		For one gene g, with w = signedWeight:

		xtb = Transpose(pooledGtx) * w
		projection = Dot(xtb, omega * xtb)

		With pooledGtx at 1/N and omega = N * (X^T X)^-1, this equals
		b^T X (X^T X)^-1 X^T b / N.
	*/
	covariateCount, _ := omega.Dims()
	xtb := mpc_core.InitRVec(mpcObj.GetRType().Zero(), covariateCount)
	publicXtb := mpcObj.SSMultMat(
		pooledGtx.Transpose(), burdenColumn(mpcObj, signedWeight),
	)
	publicXtb = mpcObj.TruncMat(
		publicXtb, mpcObj.GetDataBits(), mpcObj.GetFracBits(),
	)
	for covariate := range xtb {
		xtb[covariate] = xtb[covariate].Add(publicXtb[covariate][0])
	}

	projected := mpcObj.SSMultMat(omega, burdenColumn(mpcObj, xtb))
	projected = mpcObj.TruncMat(
		projected, mpcObj.GetDataBits(), mpcObj.GetFracBits(),
	)
	projectedVector := mpc_core.InitRVec(mpcObj.GetRType().Zero(), len(xtb))
	for covariate := range projectedVector {
		projectedVector[covariate] = projected[covariate][0]
	}
	return sharedDot(mpcObj, xtb, projectedVector)
}

func ComputeBurdenVariance(
	mpcObj *mpc.MPC,
	batch GeneBatch,
	pooledGtx []mpc_core.RMat,
	signedWeight []mpc_core.RVec,
	gtgWeight []mpc_core.RVec,
	omega mpc_core.RMat,
) mpc_core.RVec {
	/*
		For each gene g:
			V[g] = quadratic[g] - projection[g]
		V[g] remains secret-shared.
	*/
	geneBatchV := mpc_core.InitRVec(
		mpcObj.GetRType().Zero(), len(batch.GeneIndices),
	)
	for position, geneIndex := range batch.GeneIndices {
		quadratic := ComputeBurdenQuadratic(
			mpcObj, signedWeight[geneIndex], gtgWeight[position],
		)
		projection := ComputeBurdenProjectionTerm(
			mpcObj, pooledGtx[position], signedWeight[geneIndex], omega,
		)
		geneBatchV[position] = quadratic.Sub(projection)
	}
	return geneBatchV
}

func burdenColumn(mpcObj *mpc.MPC, values mpc_core.RVec) mpc_core.RMat {
	column := mpc_core.InitRMat(mpcObj.GetRType().Zero(), len(values), 1)
	for row := range values {
		column[row][0] = values[row].Copy()
	}
	return column
}
