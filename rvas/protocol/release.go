package protocol

import (
	"math"

	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
)

func Release(
	mpcObj *mpc.MPC,
	bSquared, z mpc_core.RVec,
	observe func(stage string) func(),
) (
	burdenP, skatP []float64,
) {
	/*
		Release the pivots.

		1. Reveal b² and z to every data site P1..PN (P0 receives no result).
		2. Compute p-values locally at each data site:

		burdenP = erfc(sqrt(max(b², 0)))
		skatP   = 0.5 * erfc(z / sqrt(2))

		Outputs use gene-major order g*q+t.
	*/

	// 1. Reveal the pivots.
	done := observe("open_pivots")
	openB2 := mpcObj.RevealSymVec(bSquared)
	openZ := mpcObj.RevealSymVec(z)
	done()
	if mpcObj.GetPid() == auxiliaryPartyID {
		return nil, nil
	}

	// 2. Compute the two p-values locally.
	done = observe("local_pvalues")
	fracBits := mpcObj.GetFracBits()
	burdenP = make([]float64, len(bSquared))
	skatP = make([]float64, len(bSquared))

	for index := range burdenP {
		// Fixed-point truncation can leave a b² near 0 slightly negative.
		burdenP[index] = math.Erfc(math.Sqrt(math.Max(openB2[index].Float64(fracBits), 0)))
		skatP[index] = 0.5 * math.Erfc(openZ[index].Float64(fracBits)/math.Sqrt2)
	}
	done()

	return burdenP, skatP
}
