package crypto

import (
	"fmt"
	"math"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

func EncodeDense(cryptoParams *CryptoParams, vals *mat.Dense) PlainMatrix {
	r, c := vals.Dims()
	valsEnc := make(PlainMatrix, c)
	tmp := make([]float64, r)
	for i := 0; i < c; i++ {
		valsC := mat.Col(tmp, i, vals)
		valsEnc[i], _ = EncodeFloatVector(cryptoParams, valsC)
	}
	return valsEnc
}

func EncryptDense(cryptoParams *CryptoParams, vals *mat.Dense) CipherMatrix {
	r, c := vals.Dims()
	valsEnc := make(CipherMatrix, c)
	tmp := make([]float64, r)
	for i := 0; i < c; i++ {
		valsC := mat.Col(tmp, i, vals)
		valsEnc[i], _ = EncryptFloatVector(cryptoParams, valsC)
	}
	return valsEnc
}

func PlaintextToDense(cryptoParams *CryptoParams, pt PlainMatrix, ptVecSize int) *mat.Dense {
	vals := make([]float64, len(pt)*ptVecSize)
	for i := range pt {
		tmp := DecodeFloatVector(cryptoParams, pt[i])
		for j := 0; j < ptVecSize; j++ {
			vals[i*ptVecSize+j] = tmp[j]
		}
	}
	return mat.DenseCopyOf(mat.NewDense(len(pt), ptVecSize, vals).T())
}

func EncryptPlaintextMatrix(cryptoParams *CryptoParams, pm PlainMatrix) CipherMatrix {
	cm := make(CipherMatrix, len(pm))
	if err := cryptoParams.WithEncryptor(func(encryptor *rlwe.Encryptor) error {
		for c := range pm {
			cm[c] = make(CipherVector, len(pm[c]))
			for i := range pm[c] {
				var err error
				cm[c][i], err = encryptor.EncryptNew(pm[c][i])
				if err != nil {
					return err
				}
			}
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return cm
}

// Rerandomize adds a fresh public-key encryption of zero without modifying ct
// or consuming a level.
func Rerandomize(cp *CryptoParams, ct *rlwe.Ciphertext) *rlwe.Ciphertext {
	zero := rlwe.NewCiphertext(cp.Params, 1, ct.Level())
	*zero.MetaData = *ct.MetaData
	if err := cp.WithEncryptor(func(enc *rlwe.Encryptor) error {
		return enc.EncryptZero(zero)
	}); err != nil {
		panic(err)
	}
	return Add(cp, ct, zero)
}

// RerandomizeMatrix uses independent encryption randomness for each entry.
func RerandomizeMatrix(cp *CryptoParams, cm CipherMatrix) CipherMatrix {
	out := make(CipherMatrix, len(cm))
	for i := range cm {
		out[i] = make(CipherVector, len(cm[i]))
		for j, ct := range cm[i] {
			out[i][j] = Rerandomize(cp, ct)
		}
	}
	return out
}

func GlobalToPartyIndex(cryptoParams *CryptoParams, Arowdims []int, col, nparty int) (int, int, int) {
	pid := 0
	ctid := 0
	slotid := col
	for i := 0; i < nparty; i++ {
		if Arowdims[i] > 0 && slotid < Arowdims[i] {
			pid = i
			ctid = int(slotid / cryptoParams.GetSlots())
			slotid = slotid % cryptoParams.GetSlots()
			break
		}
		slotid -= Arowdims[i]
	}
	return pid, ctid, slotid
}

func MaskTrunc(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, n int) *rlwe.Ciphertext {
	if n < 0 || n > cryptoParams.GetSlots() {
		panic(fmt.Sprintf("MaskTrunc: n=%d out of range [0, %d]", n, cryptoParams.GetSlots()))
	}

	if n == cryptoParams.GetSlots() {
		return ct
	}

	m := make([]float64, cryptoParams.GetSlots())
	for i := 0; i < n; i++ {
		m[i] = 1.0
	}

	mask, _ := EncodeFloatVector(cryptoParams, m)
	var out *rlwe.Ciphertext
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		var err error
		out, err = eval.MulRelinNew(ct, mask[0])
		if err != nil {
			return err
		}
		return eval.Rescale(out, out)
	}); err != nil {
		panic(err)
	}
	return out
}

func maskSlot(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, ind int, keep bool, scalingFactor float64, op string) *rlwe.Ciphertext {
	if ind < 0 || ind >= cryptoParams.GetSlots() {
		panic(fmt.Sprintf("%s: index=%d out of range [0, %d)", op, ind, cryptoParams.GetSlots()))
	}

	m := make([]float64, cryptoParams.GetSlots())
	if keep {
		for i := range m {
			if i != ind {
				m[i] = scalingFactor
			}
		}
	} else {
		m[ind] = scalingFactor
	}

	mask, _ := EncodeFloatVector(cryptoParams, m)
	var out *rlwe.Ciphertext
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		var err error
		out, err = eval.MulRelinNew(ct, mask[0])
		if err != nil {
			return err
		}
		return eval.Rescale(out, out)
	}); err != nil {
		panic(err)
	}
	return out
}

func MaskWithScaling(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, ind int, keep bool, scalingFactor float64) *rlwe.Ciphertext {
	return maskSlot(cryptoParams, ct, ind, keep, scalingFactor, "MaskWithScaling")
}

func Mask(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, index int, keepRest bool) *rlwe.Ciphertext {
	if ct == nil {
		return nil
	}
	return maskSlot(cryptoParams, ct, index, keepRest, 1, "Mask")
}

func Add(cryptoParams *CryptoParams, ct1 *rlwe.Ciphertext, ct2 *rlwe.Ciphertext) *rlwe.Ciphertext {
	var newct *rlwe.Ciphertext
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		var err error
		newct, err = eval.AddNew(ct1, ct2)
		return err
	}); err != nil {
		panic(err)
	}
	return newct
}

func AddConst(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, constant interface{}) *rlwe.Ciphertext {
	// Lattigo v6.1.1's scalar AddNew does not copy the input scale.
	out := ct.CopyNew()
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		return eval.Add(out, constant, out)
	}); err != nil {
		panic(err)
	}
	return out
}

func Neg(cryptoParams *CryptoParams, ct *rlwe.Ciphertext) *rlwe.Ciphertext {
	if ct == nil {
		return nil
	}
	out := ct.CopyNew()
	ringQ := cryptoParams.Params.RingQ().AtLevel(out.Level())
	for i := range out.Value {
		ringQ.Neg(out.Value[i], out.Value[i])
	}
	return out
}

func RotateRightWithEvaluator(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, nrot int, eva *ckks.Evaluator) *rlwe.Ciphertext {
	nrot = Mod(nrot, cryptoParams.GetSlots())
	if nrot == 0 {
		return ct.CopyNew()
	}
	out, err := eva.RotateNew(ct, cryptoParams.GetSlots()-nrot)
	if err != nil {
		panic(err)
	}
	return out
}

func RotateRight(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, nrot int) *rlwe.Ciphertext {
	nrot = Mod(nrot, cryptoParams.GetSlots())
	if nrot == 0 {
		return ct.CopyNew()
	}
	var out *rlwe.Ciphertext
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		var err error
		out, err = eval.RotateNew(ct, cryptoParams.GetSlots()-nrot)
		return err
	}); err != nil {
		panic(err)
	}
	return out
}

func RotateAndAdd(cryptoParams *CryptoParams, ct *rlwe.Ciphertext, size int) *rlwe.Ciphertext {
	ctOut := ct.CopyNew()
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for rotate := 1; rotate < size; rotate *= 2 {
			rt, err := eval.RotateNew(ctOut, rotate)
			if err != nil {
				return err
			}
			if err := eval.Add(ctOut, rt, ctOut); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return ctOut
}

func Rebalance(cryptoParams *CryptoParams, ct *rlwe.Ciphertext) *rlwe.Ciphertext {
	if ct == nil {
		return nil
	}
	return InnerSumAll(cryptoParams, CipherVector{ct})
}

func InnerProd(cryptoParams *CryptoParams, X, Y CipherVector) *rlwe.Ciphertext {
	return InnerSumAll(cryptoParams, CMult(cryptoParams, X, Y))
}

func InnerSumAll(cryptoParams *CryptoParams, X CipherVector) *rlwe.Ciphertext {
	slots := cryptoParams.GetSlots()
	vecsum := X[0].CopyNew()
	if len(X) > 1 {
		if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
			for i := 1; i < len(X); i++ {
				if err := eval.Add(vecsum, X[i], vecsum); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			panic(err)
		}
	}
	return RotateAndAdd(cryptoParams, vecsum, slots)
}

func SqSum(cryptoParams *CryptoParams, X CipherVector) *rlwe.Ciphertext {
	return InnerSumAll(cryptoParams, CMult(cryptoParams, X, X))
}

func Zero(cryptoParams *CryptoParams) *rlwe.Ciphertext {
	a := []float64{0.0}
	tmp, _ := EncryptFloatVector(cryptoParams, a)
	return tmp[0]
}

func CZeros(cryptoParams *CryptoParams, n int) CipherVector {
	cv := make(CipherVector, n)
	a := []float64{0.0}
	for i := range cv {
		tmp, _ := EncryptFloatVector(cryptoParams, a)
		cv[i] = tmp[0]
	}
	return cv
}

func CZeroMat(cryptoParams *CryptoParams, nrows, ncols int) CipherMatrix {
	cm := make(CipherMatrix, ncols)
	for r := range cm {
		cm[r] = CZeros(cryptoParams, nrows)
	}
	return cm
}

func CMult(cryptoParams *CryptoParams, X CipherVector, Y CipherVector) CipherVector {
	return binaryCipherVectorOp(cryptoParams, X, Y, func(eval *ckks.Evaluator, a *rlwe.Ciphertext, b rlwe.Operand) (*rlwe.Ciphertext, error) {
		out, err := eval.MulRelinNew(a, b)
		if err != nil {
			return nil, err
		}
		return out, eval.Rescale(out, out)
	})
}

func plaintextForCiphertext(cryptoParams *CryptoParams, encoder *ckks.Encoder, pt *rlwe.Plaintext, ct *rlwe.Ciphertext) (*rlwe.Plaintext, error) {
	if pt == nil || ct == nil || (pt.Level() == ct.Level() && pt.Scale.Cmp(ct.Scale) == 0) {
		return pt, nil
	}

	// Batched plaintexts can hold complex query/Fourier constants. Decoding
	// those into float64 here would silently discard their imaginary part.
	var values any = make([]float64, 1<<pt.LogDimensions.Cols)
	if pt.IsBatched {
		values = make([]complex128, 1<<pt.LogDimensions.Cols)
	}
	if err := encoder.Decode(pt, values); err != nil {
		return nil, err
	}

	out := ckks.NewPlaintext(cryptoParams.Params, ct.Level())
	*out.MetaData = *pt.MetaData
	out.Scale = ct.Scale
	if err := encoder.Encode(values, out); err != nil {
		return nil, err
	}
	return out, nil
}

func CPMult(cryptoParams *CryptoParams, X CipherVector, Y PlainVector) CipherVector {
	if len(X) == 0 || len(Y) == 0 {
		return nil
	}

	size := broadcastSize(len(X), len(Y))
	res := make(CipherVector, size)
	plainOps := make(PlainVector, size)
	if err := cryptoParams.WithEncoder(func(encoder *ckks.Encoder) error {
		for i := 0; i < size; i++ {
			x := X[minIndex(i, len(X))]
			y, err := plaintextForCiphertext(cryptoParams, encoder, Y[minIndex(i, len(Y))], x)
			if err != nil {
				return err
			}
			plainOps[i] = y
		}
		return nil
	}); err != nil {
		panic(err)
	}

	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for i := 0; i < size; i++ {
			x := X[minIndex(i, len(X))]
			y := plainOps[i]
			out, err := eval.MulRelinNew(x, y)
			if err != nil {
				return err
			}
			if err := eval.Rescale(out, out); err != nil {
				return err
			}
			res[i] = out
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return res
}

func CMultConstMat(cryptoParams *CryptoParams, X CipherMatrix, constant interface{}, inPlace bool) (res CipherMatrix) {
	res = make(CipherMatrix, len(X))
	for i := range res {
		res[i] = CMultConst(cryptoParams, X[i], constant, inPlace)
	}
	return res
}

func CMultConst(cryptoParams *CryptoParams, X CipherVector, constant interface{}, inPlace bool) (res CipherVector) {
	if inPlace {
		res = X
	} else {
		res = make(CipherVector, len(X))
	}

	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for i := range X {
			out, err := eval.MulNew(X[i], constant)
			if err != nil {
				return err
			}
			if !constantIsGaussianInteger(constant) {
				if err := eval.Rescale(out, out); err != nil {
					return err
				}
			}
			res[i] = out
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return res
}

func CMatRescale(cryptoParams *CryptoParams, X CipherMatrix) CipherMatrix {
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for j := range X {
			for i := range X[j] {
				if err := eval.Rescale(X[j][i], X[j][i]); err != nil {
					return err
				}
			}
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return X
}

func FlattenLevels(cryptoParams *CryptoParams, X CipherMatrix) (CipherMatrix, int) {
	minLevel, maxLevel := -1, -1
	for i := range X {
		for j := range X[i] {
			level := X[i][j].Level()
			if minLevel == -1 || level < minLevel {
				minLevel = level
			}
			if level > maxLevel {
				maxLevel = level
			}
		}
	}
	if minLevel == -1 {
		return X, 0
	}
	if minLevel == maxLevel {
		return X, minLevel
	}
	return DropLevel(cryptoParams, X, minLevel), minLevel
}

func constantIsGaussianInteger(constant interface{}) bool {
	switch c := constant.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return true
	case float32:
		return math.Trunc(float64(c)) == float64(c)
	case float64:
		return math.Trunc(c) == c
	case complex64:
		r, i := real(c), imag(c)
		return math.Trunc(float64(r)) == float64(r) && math.Trunc(float64(i)) == float64(i)
	case complex128:
		r, i := real(c), imag(c)
		return math.Trunc(r) == r && math.Trunc(i) == i
	case *big.Int:
		return true
	case *big.Float:
		if c == nil {
			return false
		}
		i, acc := c.Int(nil)
		if acc == big.Exact {
			return true
		}
		f := new(big.Float).SetPrec(c.Prec()).SetInt(i)
		return c.Cmp(f) == 0
	default:
		return false
	}
}

func CMultScalar(cryptoParams *CryptoParams, X CipherVector, ct *rlwe.Ciphertext) CipherVector {
	return CMult(cryptoParams, X, CipherVector{ct})
}

func CAdd(cryptoParams *CryptoParams, X CipherVector, Y CipherVector) CipherVector {
	return binaryCipherVectorOp(cryptoParams, X, Y, func(eval *ckks.Evaluator, a *rlwe.Ciphertext, b rlwe.Operand) (*rlwe.Ciphertext, error) {
		return eval.AddNew(a, b)
	})
}

func CSub(cryptoParams *CryptoParams, X CipherVector, Y CipherVector) CipherVector {
	return binaryCipherVectorOp(cryptoParams, X, Y, func(eval *ckks.Evaluator, a *rlwe.Ciphertext, b rlwe.Operand) (*rlwe.Ciphertext, error) {
		return eval.SubNew(a, b)
	})
}

func CNeg(cryptoParams *CryptoParams, X CipherVector, inPlace bool) CipherVector {
	if inPlace {
		for i := range X {
			ringQ := cryptoParams.Params.RingQ().AtLevel(X[i].Level())
			for j := range X[i].Value {
				ringQ.Neg(X[i].Value[j], X[i].Value[j])
			}
		}
		return X
	}
	res := make(CipherVector, len(X))
	for i := range X {
		res[i] = Neg(cryptoParams, X[i])
	}
	return res
}

func CPAdd(cryptoParams *CryptoParams, X CipherVector, Y PlainVector) CipherVector {
	if len(X) == 0 || len(Y) == 0 {
		return nil
	}
	size := broadcastSize(len(X), len(Y))
	res := make(CipherVector, size)
	plainOps := make(PlainVector, size)
	if err := cryptoParams.WithEncoder(func(encoder *ckks.Encoder) error {
		for i := 0; i < size; i++ {
			x := X[minIndex(i, len(X))]
			y, err := plaintextForCiphertext(cryptoParams, encoder, Y[minIndex(i, len(Y))], x)
			if err != nil {
				return err
			}
			plainOps[i] = y
		}
		return nil
	}); err != nil {
		panic(err)
	}

	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for i := 0; i < size; i++ {
			x := X[minIndex(i, len(X))]
			y := plainOps[i]
			out, err := eval.AddNew(x, y)
			if err != nil {
				return err
			}
			res[i] = out
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return res
}

func CAddConst(cryptoParams *CryptoParams, X CipherVector, constant interface{}) CipherVector {
	res := make(CipherVector, len(X))
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for i := range X {
			out, err := eval.AddNew(X[i], constant)
			if err != nil {
				return err
			}
			res[i] = out
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return res
}

func InitEncryptedMatrix(cryptoParams *CryptoParams, dy int, dx int) (CipherMatrix, int, int, error) {
	matrix := make([][]float64, dy)
	for i := range matrix {
		matrix[i] = make([]float64, dx)
	}
	return EncryptFloatMatrixRow(cryptoParams, matrix)
}

func CMask(cryptoParams *CryptoParams, cv CipherVector, index int, keepRest bool) CipherVector {
	if cv == nil {
		return nil
	}
	maxIndex := cryptoParams.GetSlots() * len(cv)
	if index < 0 || index >= maxIndex {
		panic(fmt.Sprintf("CMask: index=%d out of range [0, %d)", index, maxIndex))
	}
	m := make([]float64, cryptoParams.GetSlots()*len(cv))
	if keepRest {
		for i := range m {
			if i != index {
				m[i] = 1.0
			}
		}
	} else {
		m[index] = 1.0
	}
	mask, _ := EncodeFloatVector(cryptoParams, m)
	return CPMult(cryptoParams, cv, mask)
}

func CPSubOther(cryptoParams *CryptoParams, X PlainVector, Y CipherVector) CipherVector {
	negY := CNeg(cryptoParams, Y, false)
	return CPAdd(cryptoParams, negY, X)
}

func CRescale(cryptoParams *CryptoParams, X CipherVector) CipherVector {
	if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
		for i := range X {
			if err := eval.Rescale(X[i], X[i]); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		panic(err)
	}
	return X
}

func ConcatCipherMatrix(mats []CipherMatrix) CipherMatrix {
	nrow := -1
	for i := range mats {
		if mats[i] != nil {
			if nrow == -1 {
				nrow = len(mats[i])
			} else if len(mats[i]) != nrow {
				panic(fmt.Sprintf("ConcatCipherMatrix: matrix %d has %d rows, expected %d", i, len(mats[i]), nrow))
			}
			if len(mats[i]) > 0 {
				rowWidth := len(mats[i][0])
				for r := 1; r < len(mats[i]); r++ {
					if len(mats[i][r]) != rowWidth {
						panic(fmt.Sprintf("ConcatCipherMatrix: matrix %d row %d has %d columns, expected %d", i, r, len(mats[i][r]), rowWidth))
					}
				}
			}
		}
	}
	if nrow == -1 {
		nrow = 0
	}

	out := make(CipherMatrix, nrow)
	for r := range out {
		for _, matrix := range mats {
			if matrix != nil {
				out[r] = append(out[r], matrix[r]...)
			}
		}
	}
	return out
}

func DropLevel(cryptoParams *CryptoParams, A CipherMatrix, outLevel int) CipherMatrix {
	out := make(CipherMatrix, len(A))
	for i := range out {
		out[i] = make(CipherVector, len(A[i]))
		for j := range out[i] {
			level := A[i][j].Level()
			if level < outLevel {
				panic(fmt.Sprintf("DropLevel: requested level %d when input is %d", outLevel, level))
			}
			if level == outLevel {
				out[i][j] = A[i][j].CopyNew()
				continue
			}
			if err := cryptoParams.WithEvaluator(func(eval *ckks.Evaluator) error {
				out[i][j] = eval.DropLevelNew(A[i][j], level-outLevel)
				return nil
			}); err != nil {
				panic(err)
			}
		}
	}
	return out
}
