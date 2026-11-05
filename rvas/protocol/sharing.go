package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

const auxiliaryPartyID = 0

// ShareSum shares each data site's fixed-point matrix and sums the shares.
// P0 supplies nil and receives a zero matrix.
func ShareSum(
	mpcObj *mpc.MPC,
	localValue *mat.Dense,
	rows int,
	columns int,
) mpc_core.RMat {
	rtype := mpcObj.GetRType()
	pid := mpcObj.GetPid()

	if pid == auxiliaryPartyID {
		return mpc_core.InitRMat(rtype.Zero(), rows, columns)
	}

	sites := shareSites(mpcObj, []*mat.Dense{localValue}, [][2]int{{rows, columns}})
	sum := sites[0][0]
	for _, site := range sites[1:] {
		sum.Add(site[0])
	}
	return sum
}

func sumShares(rtype mpc_core.RElem, values mpc_core.RVec) mpc_core.RElem {
	sum := rtype.Zero()
	for _, value := range values {
		sum = sum.Add(value)
	}
	return sum
}

func sharedDot(
	mpcObj *mpc.MPC,
	left, right mpc_core.RVec,
) mpc_core.RElem {
	if len(left) == 0 {
		return mpcObj.GetRType().Zero()
	}

	products := mpcObj.SSMultElemVec(left, right)
	products = mpcObj.TruncVec(
		products,
		mpcObj.GetDataBits(),
		mpcObj.GetFracBits(),
	)
	return sumShares(mpcObj.GetRType(), products)
}
