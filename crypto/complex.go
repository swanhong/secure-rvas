package crypto

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// MulComplexSlots multiplies by local complex slot constants, encoding directly
// at ct's level and scale. Like CPMult it consumes one rescale level on PN14.
func MulComplexSlots(cp *CryptoParams, ct *rlwe.Ciphertext, values []complex128) *rlwe.Ciphertext {
	pt := ckks.NewPlaintext(cp.Params, ct.Level())
	*pt.MetaData = *ct.MetaData
	if err := cp.WithEncoder(func(enc *ckks.Encoder) error { return enc.Encode(values, pt) }); err != nil {
		panic(err)
	}
	return CPMult(cp, CipherVector{ct}, PlainVector{pt})[0]
}

// bsgsShape partitions an inclusive signed diagonal interval into power-of-two
// baby steps and signed giant steps. The shape depends only on public bounds.
func bsgsShape(first, last int) (baby, start int) {
	if first > last {
		panic("BSGS: empty diagonal interval")
	}
	baby = 1
	for baby*baby < last-first+1 {
		baby *= 2
	}
	start = first - first%baby
	if start > first {
		start -= baby
	}
	return
}

// BSGSRotations lists right rotations in execution order, including repeated
// unit baby steps. Use its length for operation counts and deduplicate the
// corresponding Galois elements for key generation.
func BSGSRotations(first, last, slots int) []int {
	baby, start := bsgsShape(first, last)
	steps := make([]int, baby-1)
	for i := range steps {
		steps[i] = 1
	}
	for giant := start; giant <= last; giant += baby {
		if giant%slots != 0 {
			steps = append(steps, giant)
		}
	}
	return steps
}

// DiagonalBSGS evaluates sum_k diag(k)*RotateRight(ct,k) with one plaintext
// multiplication level. fill writes each diagonal in OUTPUT order. Every
// diagonal in the public inclusive interval is evaluated, including zero masks.
// At most O(sqrt(last-first+1)) baby ciphertexts are retained; masks are reused.
func DiagonalBSGS(cp *CryptoParams, ct *rlwe.Ciphertext, first, last int, fill func(int, []complex128)) *rlwe.Ciphertext {
	slots := cp.GetSlots()
	if first <= -slots || last >= slots || !ct.IsBatched || ct.Slots() != slots {
		panic("BSGS: invalid diagonal range or ciphertext layout")
	}
	baby, start := bsgsShape(first, last)
	rotated := make(CipherVector, baby)
	rotated[0] = ct
	for i := 1; i < baby; i++ {
		rotated[i] = RotateRight(cp, rotated[i-1], 1)
	}
	diagonal, mask := make([]complex128, slots), make([]complex128, slots)
	var out *rlwe.Ciphertext
	for giant := start; giant <= last; giant += baby {
		var sum *rlwe.Ciphertext
		for i := 0; i < baby; i++ {
			shift := giant + i
			if shift < first || shift > last {
				continue
			}
			clear(diagonal)
			fill(shift, diagonal)
			// R_g(mask * R_i(x)) needs mask[t] = diag[t+g].
			for s, value := range diagonal {
				mask[(s-giant+slots)%slots] = value
			}
			term := MulComplexSlots(cp, rotated[i], mask)
			if sum == nil {
				sum = term
			} else {
				sum = Add(cp, sum, term)
			}
		}
		if giant%slots != 0 {
			sum = RotateRight(cp, sum, giant)
		}
		if out == nil {
			out = sum
		} else {
			out = Add(cp, out, sum)
		}
	}
	return out
}

// BlockFourier applies the positive-sign DFT or negative-sign 1/M IDFT within
// each gene block using BSGS. It evaluates all 2*M-1 masked diagonals at depth
// one; BSGSRotations(1-M, M-1, slots) describes the required right rotations.
// Inputs are preserved and unused output slots are masked to zero.
func BlockFourier(cp *CryptoParams, ct *rlwe.Ciphertext, blockSize, geneCount int, inverse bool) *rlwe.Ciphertext {
	slots := cp.GetSlots()
	if blockSize < 1 || blockSize > slots || blockSize&(blockSize-1) != 0 || geneCount < 0 || geneCount > slots/blockSize {
		panic("BlockFourier: invalid block layout")
	}
	return DiagonalBSGS(cp, ct, 1-blockSize, blockSize-1, func(shift int, mask []complex128) {
		for s := 0; s < geneCount*blockSize; s++ {
			local := s % blockSize
			t := local - shift
			if t < 0 || t >= blockSize {
				continue
			}
			angle := 2 * math.Pi * float64(local*t) / float64(blockSize)
			if inverse {
				mask[s] = cmplx.Rect(1/float64(blockSize), -angle)
			} else {
				mask[s] = cmplx.Rect(1, angle)
			}
		}
	})
}

// EncodeComplexVector packs values at the maximum level and default scale.
// As with EncodeFloatVector, the last plaintext is padded to the slot count.
func EncodeComplexVector(cp *CryptoParams, values []complex128) (PlainVector, int) {
	return EncodeComplexVectorWithScale(cp, values, cp.Params.DefaultScale().Float64())
}

// EncodeComplexVectorWithScale packs values at the maximum level and given scale.
// Complex values require the standard CKKS ring used by the embedding parameters.
func EncodeComplexVectorWithScale(cp *CryptoParams, values []complex128, scale float64) (PlainVector, int) {
	slots := cp.GetSlots()
	plain := make(PlainVector, 0, (len(values)+slots-1)/slots)
	for start := 0; start < len(values); start += slots {
		end := Min(start+slots, len(values))
		pt := ckks.NewPlaintext(cp.Params, cp.Params.MaxLevel())
		pt.Scale = rlwe.NewScale(scale)
		if err := cp.WithEncoder(func(encoder *ckks.Encoder) error {
			return encoder.Encode(values[start:end], pt)
		}); err != nil {
			panic(err)
		}
		plain = append(plain, pt)
	}
	return plain, len(values)
}

// EncryptComplexVector encrypts values under cp's public key at the maximum
// level and default scale, returning the number of input values.
func EncryptComplexVector(cp *CryptoParams, values []complex128) (CipherVector, int) {
	plain, n := EncodeComplexVector(cp, values)
	return EncryptPlaintextMatrix(cp, PlainMatrix{plain})[0], n
}

// DecodeComplexVector decodes each plaintext into a full block of slots,
// retaining imaginary values and any final-block padding.
func DecodeComplexVector(cp *CryptoParams, plain PlainVector) []complex128 {
	decoded := make([]complex128, 0, len(plain)*cp.GetSlots())
	for _, pt := range plain {
		values := make([]complex128, cp.GetSlots())
		if err := cp.WithEncoder(func(encoder *ckks.Encoder) error {
			return encoder.Decode(pt, values)
		}); err != nil {
			panic(err)
		}
		decoded = append(decoded, values...)
	}
	return decoded
}

// BlockSumBroadcast replaces every slot of each of geneCount consecutive
// blocks with that block's sum. Other slots are masked to zero. The ciphertext
// uses the full slot count, and blockSize is a power of two dividing it.
//
// The sum uses left rotations, the head mask retains slots g*blockSize, and
// right rotations broadcast each head within its block. The mask is multiplied
// at the input scale and rescaled, exactly as CPMult: PN14 consumes one level
// and leaves scale approximately inputScale^2/Q[inputLevel]. Rotations/additions
// consume no levels. The input ciphertext is not modified.
func BlockSumBroadcast(cp *CryptoParams, ct *rlwe.Ciphertext, blockSize, geneCount int) *rlwe.Ciphertext {
	slots := cp.GetSlots()
	if blockSize < 1 || blockSize > slots || blockSize&(blockSize-1) != 0 {
		panic(fmt.Sprintf("BlockSumBroadcast: invalid block size %d for %d slots", blockSize, slots))
	}
	if geneCount < 0 || geneCount > slots/blockSize {
		panic(fmt.Sprintf("BlockSumBroadcast: invalid gene count %d for block size %d", geneCount, blockSize))
	}
	if !ct.IsBatched || ct.Slots() != slots {
		panic("BlockSumBroadcast: ciphertext must use the full batched slot count")
	}

	out := RotateAndAdd(cp, ct, blockSize)
	heads := make([]float64, slots)
	for g := 0; g < geneCount; g++ {
		heads[g*blockSize] = 1
	}
	mask, _ := EncodeFloatVector(cp, heads)
	out = CPMult(cp, CipherVector{out}, mask)[0]
	for step := 1; step < blockSize; step *= 2 {
		out = Add(cp, out, RotateRight(cp, out, step))
	}
	return out
}
