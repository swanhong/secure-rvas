package embed

import (
	"math"
	"math/cmplx"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func (c *tableCircuit) appendHashStash(shape HashShape, genes int, state, active, proposed crypto.CipherVector, keys [][]string) {
	cp := c.cp
	maxLevel := cp.Params.MaxLevel()
	if shape.Sigma == 0 {
		return
	}

	r, k := shape.Rho+shape.Sigma, shape.stashTransform()
	var frontier *rlwe.Ciphertext
	var omega []complex128
	if c.owner() {
		stash := make([]complex128, cp.GetSlots())
		omega = make([]complex128, cp.GetSlots())
		for g := 0; g < genes; g++ {
			for s := 0; s < shape.Sigma; s++ {
				stash[g*r+s] = 1
			}
			for s := 0; s < k; s++ {
				omega[g*r+s] = cmplx.Rect(1, 2*math.Pi*float64(s)/float64(k)) - 1
			}
		}
		x := c.plain(c.rotate(state[planeO], -shape.Rho), stash)
		frontier = crypto.AddConst(cp, c.plain(c.fourier(x, k, genes*(r/k), false), omega), 1)
	}
	frontier = c.refresh(crypto.CipherVector{frontier}, 1)[0]
	var prefix crypto.CipherVector
	if c.owner() {
		prefix = make(crypto.CipherVector, shape.M)
		prefix[0] = frontier
		for j := 1; j < shape.M; j++ {
			prefix[j] = crypto.AddConst(cp, c.plain(active[j-1], omega), 1)
		}
		prefix = crypto.DropLevel(cp, crypto.CipherMatrix{prefix}, maxLevel-1)[0]
	}
	prefix = c.inclusiveProducts(prefix, shape.M, maxLevel-1)
	var writes crypto.CipherVector
	if c.owner() {
		writes = make(crypto.CipherVector, shape.M)
		for j := range writes {
			writes[j] = c.fourier(c.mul(active[j], prefix[j]), k, genes*(r/k), true)
		}
		c.record("stash IDFT selectors", writes)
	}
	writes = c.refresh(writes, shape.M)
	if c.owner() {
		encoded := encodeLocalKeys(keys)
		codes := make([]complex128, cp.GetSlots())
		stashMask := make([]complex128, cp.GetSlots())
		for g := 0; g < genes; g++ {
			for s := shape.Rho; s < r; s++ {
				stashMask[g*r+s] = 1
			}
		}
		for j, w := range writes {
			w = c.rotate(w, shape.Rho)
			state[planeO] = crypto.Add(cp, state[planeO], c.plain(w, stashMask))
			state[planeD] = crypto.Add(cp, state[planeD], c.plain(c.mul(w, proposed[j]), stashMask))
			for d := 0; d < planeO; d++ {
				clear(codes)
				for g, local := range encoded {
					if j < len(local) {
						x := local[j][d]
						for s := shape.Rho; s < r; s++ {
							codes[g*r+s] = x
						}
					}
				}
				state[d] = crypto.Add(cp, state[d], c.plain(w, codes))
			}
		}
	}
}
