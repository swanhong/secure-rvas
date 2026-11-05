package crypto

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func Max(x, y int) int {
	if x <= y {
		return y
	}
	return x
}

func Min(a int, b int) int {
	if a > b {
		return b
	}
	return a
}

func Mod(n int, modulus int) int {
	n = n % modulus
	if n < 0 {
		n += modulus
	}
	return n
}

func binaryCipherVectorOp(cryptoParams *CryptoParams, X CipherVector, Y CipherVector, op func(eval *ckks.Evaluator, a *rlwe.Ciphertext, b rlwe.Operand) (*rlwe.Ciphertext, error)) CipherVector {
	if len(X) == 0 || len(Y) == 0 {
		return nil
	}
	size := broadcastSize(len(X), len(Y))
	res := make(CipherVector, size)
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for i := 0; i < size; i++ {
			var err error
			res[i], err = op(eval, X[minIndex(i, len(X))], Y[minIndex(i, len(Y))])
			if err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return res
}

func minIndex(i, length int) int {
	if length <= 1 {
		return 0
	}
	return i
}

func broadcastSize(xlen, ylen int) int {
	if xlen == ylen {
		return xlen
	}
	if xlen == 1 {
		return ylen
	}
	if ylen == 1 {
		return xlen
	}
	panic(fmt.Sprintf("incompatible vector lengths for broadcast: %d and %d", xlen, ylen))
}
