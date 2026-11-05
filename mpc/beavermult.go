package mpc

import (
	"math/big"
	"runtime"
	"sync"

	mpc_core "github.com/hhcho/mpc-core"
)

func parMultMat(a, b mpc_core.RMat) mpc_core.RMat {
	rtype := a.Type()
	r1, c1 := a.Dims()
	_, c2 := b.Dims()
	var out mpc_core.RMat
	var row func(int)
	if _, ok := rtype.(mpc_core.LElem256); ok {
		out = make(mpc_core.RMat, r1)
		for i := range out {
			out[i] = make(mpc_core.RVec, c2)
		}
		// Use the compact fixed-limb cache for large public shapes. Small
		// products retain the pointer-only big.Int path and its low setup cost.
		if int64(r1)*int64(c2)*int64(c1) >= 1<<18 && mpc_core.LElem256Mod.Val.BitLen() <= 256 {
			row = lelem256FixedMatRows(a, b, out)
		} else {
			row = lelem256MatRows(a, b, out)
		}
	} else {
		out = mpc_core.InitRMat(rtype.Zero(), r1, c2)
		row = func(i int) {
			for j := 0; j < c2; j++ {
				for k := 0; k < c1; k++ {
					out[i][j] = out[i][j].Add(a[i][k].Mul(b[k][j]))
				}
			}
		}
	}
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || r1 < 2*nw || int64(r1)*int64(c2)*int64(c1) < 1<<18 {
		for i := 0; i < r1; i++ {
			row(i)
		}
		return out
	}
	var wg sync.WaitGroup
	chunk := (r1 + nw - 1) / nw
	for lo := 0; lo < r1; lo += chunk {
		hi := lo + chunk
		if hi > r1 {
			hi = r1
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				row(i)
			}
		}(lo, hi)
	}
	wg.Wait()
	return out
}

// Integer accumulation followed by one reduction equals the field's modular
// Mul/Add dot product. Scratch belongs to each row call, inputs remain read-only,
// and every public inner position is processed, including zero-valued entries.
func lelem256MatRows(a, b, out mpc_core.RMat) func(int) {
	r1, inner := a.Dims()
	_, columns := b.Dims()
	left := make([][]*big.Int, r1)
	for i := range left {
		left[i] = make([]*big.Int, inner)
		for k := range left[i] {
			left[i][k] = a[i][k].(mpc_core.LElem256).Val
		}
	}
	right := make([][]*big.Int, columns)
	for j := range right {
		right[j] = make([]*big.Int, inner)
		for k := range right[j] {
			right[j][k] = b[k][j].(mpc_core.LElem256).Val
		}
	}
	return func(i int) {
		var product, accumulator big.Int
		for j, column := range right {
			accumulator.SetInt64(0)
			for k, value := range left[i] {
				product.Mul(value, column[k])
				accumulator.Add(&accumulator, &product)
			}
			out[i][j] = mpc_core.LElem256{Val: new(big.Int).Mod(&accumulator, mpc_core.LElem256Mod.Val)}
		}
	}
}

func (mpcObj *MPC) BeaverPartition(a mpc_core.RElem) (mpc_core.RElem, mpc_core.RElem) {
	ar, am := mpcObj.BeaverPartitionMat(mpc_core.RMat{mpc_core.RVec{a}})
	return ar[0][0], am[0][0]
}

func (mpcObj *MPC) BeaverPartitionVec(a mpc_core.RVec) (mpc_core.RVec, mpc_core.RVec) {
	ar, am := mpcObj.BeaverPartitionMat(mpc_core.RMat{a})
	return ar[0], am[0]
}

// Returns ar, am
func (mpcObj *MPC) BeaverPartitionMat(a mpc_core.RMat) (mpc_core.RMat, mpc_core.RMat) {
	pid := mpcObj.Network.pid
	nrows, ncols := a.Dims()
	rtype := a.Type()

	var mask mpc_core.RMat

	if pid == 0 {
		am := mpc_core.InitRMat(rtype.Zero(), nrows, ncols)

		for p := 1; p < mpcObj.Network.NumParties; p++ {
			mpcObj.Network.Rand.SwitchPRG(p)
			mask = mpcObj.Network.Rand.RandMat(rtype, nrows, ncols)
			am.Add(mask)
			mpcObj.Network.Rand.RestorePRG()
		}

		return mpc_core.InitRMat(rtype.Zero(), nrows, ncols), am
	}

	// Retrieve random shares from pid = 0
	mpcObj.Network.Rand.SwitchPRG(0)
	mask = mpcObj.Network.Rand.RandMat(rtype, nrows, ncols)
	mpcObj.Network.Rand.RestorePRG()

	ar := a.Copy()
	ar.Sub(mask)
	ar = mpcObj.RevealSymMat(ar)
	return ar, mask
}

func (mpcObj *MPC) BeaverReconstruct(a mpc_core.RElem) mpc_core.RElem {
	return mpcObj.BeaverReconstructMat(mpc_core.RMat{mpc_core.RVec{a}})[0][0]
}

func (mpcObj *MPC) BeaverReconstructVec(a mpc_core.RVec) mpc_core.RVec {
	return mpcObj.BeaverReconstructMat(mpc_core.RMat{a})[0]
}

func (mpcObj *MPC) BeaverReconstructMat(a mpc_core.RMat) mpc_core.RMat {
	pid := mpcObj.Network.pid

	rtype := a.Type()
	nr, nc := a.Dims()

	last := mpcObj.Network.NumParties - 1

	if pid == 0 {

		mask := a.Copy()
		for to := 1; to < mpcObj.Network.NumParties-1; to++ {
			mpcObj.Network.Rand.SwitchPRG(to)
			share := mpcObj.Network.Rand.RandMat(rtype, nr, nc)
			mpcObj.Network.Rand.RestorePRG()
			mask.Sub(share)
		}
		mpcObj.Network.SendRData(mask, last)
		return mask //return a

	}

	var mask mpc_core.RMat
	if pid == last {
		mask = mpcObj.Network.ReceiveRMat(rtype, nr, nc, 0)
	} else {
		mpcObj.Network.Rand.SwitchPRG(0)
		mask = mpcObj.Network.Rand.RandMat(rtype, nr, nc)
		mpcObj.Network.Rand.RestorePRG()
	}

	ar := a.Copy()
	ar.Add(mask)

	return ar
}

func (mpcObj *MPC) BeaverMult(ar, am, br, bm mpc_core.RElem) mpc_core.RElem {
	pid := mpcObj.Network.pid
	if pid == 0 {
		return am.Mul(bm)
	}

	out := ar.Mul(bm)
	out = out.Add(br.Mul(am))
	if pid == 1 {
		out = out.Add(ar.Mul(br))
	}
	return out
}

func (mpcObj *MPC) BeaverMultElemVec(ar, am, br, bm mpc_core.RVec) mpc_core.RVec {
	return mpcObj.BeaverMultElemMat(mpc_core.RMat{ar}, mpc_core.RMat{am}, mpc_core.RMat{br}, mpc_core.RMat{bm})[0]
}

func (mpcObj *MPC) BeaverMultElemMat(ar, am, br, bm mpc_core.RMat) mpc_core.RMat {
	pid := mpcObj.Network.pid
	nr, nc := am.Dims()

	if pid == 0 {
		out := am.Copy()
		out.MulElem(bm)
		return out
	}

	out := mpc_core.InitRMat(am.Type().Zero(), nr, nc)
	for i := 0; i < nr; i++ {
		for j := 0; j < nc; j++ {
			out[i][j] = out[i][j].Add(ar[i][j].Mul(bm[i][j]))
			out[i][j] = out[i][j].Add(br[i][j].Mul(am[i][j]))
			if pid == 1 {
				out[i][j] = out[i][j].Add(ar[i][j].Mul(br[i][j]))
			}
		}
	}
	return out
}

func (mpcObj *MPC) BeaverMultMat(ar, am, br, bm mpc_core.RMat) mpc_core.RMat {
	pid := mpcObj.Network.pid
	if pid == 0 {
		return parMultMat(am, bm)
	}

	out := parMultMat(ar, bm)
	out.Add(parMultMat(am, br))
	if pid == 1 {
		out.Add(parMultMat(ar, br))
	}
	return out
}
