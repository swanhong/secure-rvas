package protocol

import (
	"math/rand"

	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

func TraceProbe(
	capM int,
	probeCount int,
	seed int64,
) (
	traceProbe *mat.Dense,
	scale float64,
) {
	/*
		Exact (capM ≤ probeCount):
		    probe = Identity(capM)
		    scale = 1

		Hutchinson:
		    probe = deterministic Rademacher matrix
		    scale = 1 / probeCount
	*/
	if capM <= probeCount {
		identity := mat.NewDense(
			capM,
			capM,
			nil,
		)
		for variant := 0; variant < capM; variant++ {
			identity.Set(variant, variant, 1)
		}
		return identity, 1
	}

	scale = 1 / float64(probeCount)
	rng := rand.New(rand.NewSource(seed))
	values := make([]float64, capM*probeCount)
	for index := range values {
		values[index] = -1
		if rng.Intn(2) == 1 {
			values[index] = 1
		}
	}
	return mat.NewDense(capM, probeCount, values), scale
}

func scaleSharedRows(
	mpcObj *mpc.MPC,
	weight mpc_core.RVec,
	matrix mpc_core.RMat,
) mpc_core.RMat {
	/*
		Multiply each matrix row by the corresponding shared weight.

		scaled = Diag(weight) * matrix
	*/
	rows, columns := matrix.Dims()
	rowWeight := mpc_core.InitRMat(
		mpcObj.GetRType().Zero(),
		rows,
		columns,
	)

	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			rowWeight[row][column] = weight[row].Copy()
		}
	}

	scaled := mpcObj.SSMultElemMat(rowWeight, matrix)
	return mpcObj.TruncMat(
		scaled,
		mpcObj.GetDataBits(),
		mpcObj.GetFracBits(),
	)
}

func ComputeKppRight(
	mpcObj *mpc.MPC,
	weightedRight mpc_core.RMat,
	gtgRight mpc_core.RMat,
	pooledGtx mpc_core.RMat,
	theta mpc_core.RMat,
	weight mpc_core.RVec,
) mpc_core.RMat {
	/*
		Compute Kpp * right from inputs:
			weightedRight = Diag(weight) * right
		    gtgRight      = pooledGtG * weightedRight (pooledGtG is already divided by N)
		    pooledGtx     = Transpose(Gp) * X / N
		    theta         = pooledGtx * Omega'

		1. Remove the covariate projection:
		    mppRight
		        = gtgRight
		        - theta * Transpose(pooledGtx) * weightedRight

		2. Apply the outer weight:
		    KppRight
		        = 0.5 * Diag(weight) * mppRight

		Output:
		    KppRight = Kpp * right
	*/
	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()

	// 1. Remove the covariate projection.
	//   mppRight = gtgRight - theta * Transpose(pooledGtx) * weightedRight
	covariateRight := mpcObj.SSMultMat(pooledGtx.Transpose(), weightedRight)
	covariateRight = mpcObj.TruncMat(covariateRight, dataBits, fracBits)

	projection := mpcObj.SSMultMat(theta, covariateRight)
	projection = mpcObj.TruncMat(projection, dataBits, fracBits)

	mppRight := gtgRight.Copy()
	mppRight.Sub(projection)

	// 2. Apply the outer weight and the factor 0.5.
	//  kppRight = 0.5 * Diag(weight) * mppRight
	kppRight := scaleSharedRows(mpcObj, weight, mppRight)
	kppRight.MulScalar(mpcObj.GetRType().FromFloat64(0.5, fracBits))

	return mpcObj.TruncMat(kppRight, dataBits, fracBits)
}
