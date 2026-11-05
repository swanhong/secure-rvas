package cap

import (
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
)

// P0 is the helper; data sites are P1..PN, with P1 as hub/anchor.
const (
	helperPID = 0
	party1PID = 1
)

// DeriveHashKey delegates to the shared MPC input layer; P0 gets nil.
func DeriveHashKey(mpcObj *mpc.MPC) []byte {
	return mpcObj.DeriveHashKey()
}

// ShareFromOwner delegates Share_k to the shared MPC input layer.
func ShareFromOwner(mpcObj *mpc.MPC, owner int, x mpc_core.RVec, n int) mpc_core.RVec {
	return mpcObj.ShareFromOwner(owner, x, n)
}

// positivePublicShift returns shares of I[x_i − shift_i > 0] for unscaled integers x. The public
// shift is subtracted at the hub share only; field IsPositive is [x > 0] (pinned by TestSecureCaps).
func positivePublicShift(mpcObj *mpc.MPC, x mpc_core.RVec, shift []int64) mpc_core.RVec {
	rtype := mpcObj.GetRType()
	diff := x.Copy()
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		for i := range diff {
			diff[i] = diff[i].Sub(rtype.FromInt(int(shift[i])))
		}
	}
	return mpcObj.IsPositive(diff, false)
}

// mergeEmpty returns shares of V = Σ_ζ ∏_site (1-b_site,ζ) per gene:
// a bucket is empty in the union exactly when it is empty at every site.
func mergeEmpty(mpcObj *mpc.MPC, p Params, geneCount int, bitmaps [][]uint8) mpc_core.RVec {
	pid, rtype := mpcObj.GetPid(), mpcObj.GetRType()
	mu := p.Mu()
	n := geneCount * mu

	var local mpc_core.RVec
	if pid != helperPID {
		local = mpc_core.InitRVec(rtype.Zero(), n)
		for g, bitmap := range bitmaps {
			for k, bit := range bitmap {
				if bit == 1 {
					local[g*mu+k] = rtype.One()
				}
			}
		}
	}
	factors := make([]mpc_core.RVec, 0, mpcObj.GetNParty()-1)
	for site := party1PID; site < mpcObj.GetNParty(); site++ {
		bits := ShareFromOwner(mpcObj, site, local, n)
		factors = append(factors, mpcObj.FlipBit(bits))
	}
	// One batched multiplication per tree layer: N-1 products per bucket
	// in ceil(log2(N)) layers. An unpaired factor carries to the next layer.
	for len(factors) > 1 {
		var left, right mpc_core.RVec
		for f := 0; f+1 < len(factors); f += 2 {
			left = append(left, factors[f]...)
			right = append(right, factors[f+1]...)
		}
		products := mpcObj.SSMultElemVec(left, right)
		next := make([]mpc_core.RVec, 0, (len(factors)+1)/2)
		for f := 0; f < len(factors)/2; f++ {
			next = append(next, products[f*n:(f+1)*n])
		}
		if len(factors)%2 == 1 {
			next = append(next, factors[len(factors)-1])
		}
		factors = next
	}
	empty := mpc_core.InitRVec(rtype.Zero(), geneCount)
	if pid == helperPID {
		return empty
	}
	for i, bit := range factors[0] {
		empty[i/mu] = empty[i/mu].Add(bit)
	}
	return empty
}

// SecureCaps returns the public ladder index of each of geneCount genes at every party. Each data
// site passes one bitmap of length μ per gene; P0 passes nil. Only the indices are revealed.
func SecureCaps(mpcObj *mpc.MPC, p Params, th Thresholds, geneCount int, bitmaps [][]uint8) []int {
	if mpcObj.GetHubPid() != party1PID {
		panic("cap: SecureCaps requires hub pid 1") // IsPositive adds its constants at pid 1
	}
	if p.J < 1 {
		panic("cap: SecureCaps requires a ladder with J ≥ 1")
	}
	idx := make([]int, geneCount)
	if geneCount == 0 {
		return idx // public shapes, so every party returns here together
	}
	empty := mergeEmpty(mpcObj, p, geneCount, bitmaps)

	// χ_i = [V ≤ TV_i] = 1 − [V − TV_i > 0] for every gene and rung in one call; idx = Σ_i χ_i.
	gj := geneCount * p.J
	values, shifts := make(mpc_core.RVec, gj), make([]int64, gj)
	for k := range values {
		values[k], shifts[k] = empty[k/p.J], th.TV[k%p.J]
	}
	chi := mpcObj.FlipBit(positivePublicShift(mpcObj, values, shifts))
	idxShares := mpc_core.InitRVec(mpcObj.GetRType().Zero(), geneCount)
	for k := range chi {
		idxShares[k/p.J] = idxShares[k/p.J].Add(chi[k])
	}

	// Reveal idx among all data sites; P1 forwards it to P0.
	var revealed []uint64
	if mpcObj.GetPid() == helperPID {
		revealed = mpcObj.Network.ReceiveIntVector(geneCount, party1PID)
	} else {
		revealed = make([]uint64, geneCount)
		for g, v := range mpcObj.RevealSymVec(idxShares) {
			revealed[g] = v.Uint64()
		}
		if mpcObj.GetPid() == party1PID {
			mpcObj.Network.SendIntVector(revealed, helperPID)
		}
	}
	for g, v := range revealed {
		idx[g] = int(v)
	}
	return idx
}
