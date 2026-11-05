package mpc

import (
	"math/big"
	"math/bits"

	mpc_core "github.com/hhcho/mpc-core"
)

type lelem256Limbs [4]uint64

// The public inner dimension is an int. Nine 64-bit limbs hold the sum of
// MaxInt products of two unsigned 256-bit values, even on a 64-bit machine:
// each product is < 2^512 and MaxInt < 2^63, so the sum is < 2^575.
type lelem256Accumulator [9]uint64

func lelem256FixedMatRows(a, b, out mpc_core.RMat) func(int) {
	rows, inner := a.Dims()
	_, columns := b.Dims()
	left := make([][]lelem256Limbs, rows)
	right := make([][]lelem256Limbs, columns)
	var reduced big.Int
	load := func(value mpc_core.RElem) lelem256Limbs {
		// Reduction once per input preserves negative/noncanonical values and
		// never mutates the original field element or shared modulus.
		reduced.Mod(value.(mpc_core.LElem256).Val, mpc_core.LElem256Mod.Val)
		var limbs lelem256Limbs
		words := reduced.Bits()
		if bits.UintSize == 64 {
			for i := range limbs {
				if i < len(words) {
					limbs[i] = uint64(words[i])
				}
			}
		} else {
			for i := range limbs {
				if 2*i < len(words) {
					limbs[i] = uint64(words[2*i])
				}
				if 2*i+1 < len(words) {
					limbs[i] |= uint64(words[2*i+1]) << 32
				}
			}
		}
		return limbs
	}
	for i := range left {
		left[i] = make([]lelem256Limbs, inner)
		for k := range left[i] {
			left[i][k] = load(a[i][k])
		}
	}
	for j := range right {
		right[j] = make([]lelem256Limbs, inner)
		for k := range right[j] {
			right[j][k] = load(b[k][j])
		}
	}
	return func(i int) {
		var words [18]big.Word
		var integer big.Int
		for j, column := range right {
			var accumulator lelem256Accumulator
			for k := range left[i] {
				lelem256Accumulate(&accumulator, &left[i][k], &column[k])
			}
			wordCount := len(accumulator)
			if bits.UintSize == 64 {
				for k, limb := range accumulator {
					words[k] = big.Word(limb)
				}
			} else {
				wordCount *= 2
				for k, limb := range accumulator {
					words[2*k], words[2*k+1] = big.Word(uint32(limb)), big.Word(uint32(limb>>32))
				}
			}
			integer.SetBits(words[:wordCount])
			// This independent receiver owns its result; no output aliases the
			// reusable accumulator, input values, another cell, or the modulus.
			out[i][j] = mpc_core.LElem256{Val: new(big.Int).Mod(&integer, mpc_core.LElem256Mod.Val)}
		}
	}
}

// lelem256Accumulate evaluates every limb product and carry position. Its loop
// counts do not depend on zero values, sign, candidate occupancy, or key data.
func lelem256Accumulate(acc *lelem256Accumulator, x, y *lelem256Limbs) {
	var product [8]uint64
	for i := 0; i < 4; i++ {
		var carry uint64
		for j := 0; j < 4; j++ {
			hi, lo := bits.Mul64(x[j], y[i])
			var bit uint64
			lo, bit = bits.Add64(lo, product[i+j], 0)
			hi, _ = bits.Add64(hi, 0, bit)
			lo, bit = bits.Add64(lo, carry, 0)
			hi, _ = bits.Add64(hi, 0, bit)
			product[i+j], carry = lo, hi
		}
		product[i+4] = carry
	}
	var carry uint64
	for i := range product {
		acc[i], carry = bits.Add64(acc[i], product[i], carry)
	}
	acc[8], _ = bits.Add64(acc[8], 0, carry)
}
