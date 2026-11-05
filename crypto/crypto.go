package crypto

import (
	"bytes"
	"encoding"
	"encoding/gob"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strconv"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type IntervalApprox struct {
	A          float64
	B          float64
	Degree     int
	Iter       int
	InverseNew bool
}

type CipherVector []*rlwe.Ciphertext
type CipherMatrix []CipherVector
type PlainVector []*rlwe.Plaintext
type PlainMatrix []PlainVector

type RotationType struct {
	Value int
	Side  bool
}

var SideRight = true
var SideLeft = false

type CryptoParams struct {
	Sk          *rlwe.SecretKey
	AggregateSk *rlwe.SecretKey
	Pk          *rlwe.PublicKey
	Rlk         *rlwe.RelinearizationKey
	RotKs       []*rlwe.GaloisKey
	Params      ckks.Parameters

	encoders   chan *ckks.Encoder
	encryptors chan *rlwe.Encryptor
	decryptors chan *rlwe.Decryptor
	evaluators chan *ckks.Evaluator

	numThreads int
	prec       uint
}

type cryptoParamsMarshalable struct {
	Params      []byte
	Sk          []byte
	AggregateSk []byte
	Pk          []byte
	Rlk         []byte
	RotKs       [][]byte
	NumThreads  int
	Prec        uint
}

func NewCryptoParams(params ckks.Parameters, sk, aggregateSk *rlwe.SecretKey, pk *rlwe.PublicKey, rlk *rlwe.RelinearizationKey, prec uint, numThreads int) *CryptoParams {
	if numThreads <= 0 {
		numThreads = runtime.GOMAXPROCS(0)
	}

	cp := &CryptoParams{
		Sk:          sk,
		AggregateSk: aggregateSk,
		Pk:          pk,
		Rlk:         rlk,
		Params:      params,
		numThreads:  numThreads,
		prec:        prec,
	}
	cp.rebuildPools()
	return cp
}

func (cp *CryptoParams) rebuildPools() {
	cp.encoders = make(chan *ckks.Encoder, cp.numThreads)
	cp.encryptors = make(chan *rlwe.Encryptor, cp.numThreads)
	cp.decryptors = make(chan *rlwe.Decryptor, cp.numThreads)
	cp.evaluators = make(chan *ckks.Evaluator, cp.numThreads)

	var wg sync.WaitGroup
	for i := 0; i < cp.numThreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cp.encoders <- ckks.NewEncoder(cp.Params, cp.prec)
		}()
	}
	wg.Wait()

	for i := 0; i < cp.numThreads; i++ {
		cp.encryptors <- ckks.NewEncryptor(cp.Params, cp.Pk)
		cp.decryptors <- ckks.NewDecryptor(cp.Params, cp.AggregateSk)
		cp.evaluators <- ckks.NewEvaluator(cp.Params, cp.evaluationKeySet())
	}
}

func (cp *CryptoParams) evaluationKeySet() *rlwe.MemEvaluationKeySet {
	return rlwe.NewMemEvaluationKeySet(cp.Rlk, cp.RotKs...)
}

func (cp *CryptoParams) SetEvaluators(params ckks.Parameters, rlk *rlwe.RelinearizationKey, rotKs []*rlwe.GaloisKey) {
	cp.Params = params
	cp.Rlk = rlk
	cp.RotKs = rotKs
	cp.rebuildPools()
}

func (cp *CryptoParams) SetRotKeys(nbrRot []RotationType) []int {
	kgen := ckks.NewKeyGenerator(cp.Params)
	ks := make([]int, 0, len(nbrRot))
	seen := make(map[int]struct{}, len(nbrRot))

	for _, rot := range nbrRot {
		rotation := rot.Value
		if rot.Side == SideRight {
			rotation = cp.GetSlots() - rot.Value
		}
		rotation = Mod(rotation, cp.GetSlots())
		if _, ok := seen[rotation]; !ok {
			seen[rotation] = struct{}{}
			ks = append(ks, rotation)
		}
	}

	sort.Ints(ks)
	cp.RotKs = make([]*rlwe.GaloisKey, 0, len(ks))
	for _, rotation := range ks {
		cp.RotKs = append(cp.RotKs, kgen.GenGaloisKeyNew(cp.Params.GaloisElementForRotation(rotation), cp.AggregateSk))
	}
	cp.rebuildPools()

	return ks
}

func GenerateRotKeys(slots int, smallDim int, babyFlag bool) []RotationType {
	rotations := make([]RotationType, 0)
	for rot := 1; rot < slots; rot *= 2 {
		rotations = append(rotations, RotationType{Value: rot, Side: false})
		rotations = append(rotations, RotationType{Value: rot, Side: true})
	}

	if babyFlag {
		rootl := 0
		for rootl*rootl < slots {
			rootl++
		}
		for i := 1; i < rootl; i++ {
			rotations = append(rotations, RotationType{Value: i, Side: false})
			rotations = append(rotations, RotationType{Value: i * rootl, Side: false})
		}
	}

	for i := 1; i < smallDim; i++ {
		rotations = append(rotations, RotationType{Value: i, Side: true})
	}

	return rotations
}

func (cp *CryptoParams) GetPrec() uint {
	return cp.prec
}

func (cp *CryptoParams) GetSlots() int {
	return cp.Params.MaxSlots()
}

func (cp *CryptoParams) WithEncoder(act func(*ckks.Encoder) error) error {
	encoder := <-cp.encoders
	err := act(encoder)
	cp.encoders <- encoder
	return err
}

func (cp *CryptoParams) WithEncryptor(act func(*rlwe.Encryptor) error) error {
	encryptor := <-cp.encryptors
	err := act(encryptor)
	cp.encryptors <- encryptor
	return err
}

func (cp *CryptoParams) WithDecryptor(act func(*rlwe.Decryptor) error) error {
	decryptor := <-cp.decryptors
	err := act(decryptor)
	cp.decryptors <- decryptor
	return err
}

func (cp *CryptoParams) WithEvaluator(act func(*ckks.Evaluator) error) error {
	eval := <-cp.evaluators
	err := act(eval)
	cp.evaluators <- eval
	return err
}

// #------------------------------------#
// #------------ ENCRYPTION ------------#
// #------------------------------------#
// EncryptFloatVector encrypts a slice of float64 values in multiple batched ciphertexts.
// and return the number of encrypted elements.
func EncryptFloatVector(cryptoParams *CryptoParams, f []float64) (CipherVector, int) {
	plainArr, elementsEncoded := EncodeFloatVector(cryptoParams, f)
	cipherArr := make(CipherVector, len(plainArr))

	for i := range plainArr {
		if err := cryptoParams.WithEncryptor(func(encryptor *rlwe.Encryptor) error {
			var err error
			cipherArr[i], err = encryptor.EncryptNew(plainArr[i])
			return err
		}); err != nil {
			panic(err)
		}
	}

	return cipherArr, elementsEncoded
}

// EncryptFloatMatrixRow encrypts a matrix of float64 to multiple packed ciphertexts.
// For this specific matrix encryption each row is encrypted in a set of ciphertexts.
// It returns the encrypted matrix, the number of rows, the number of columns, and an error if any.
func EncryptFloatMatrixRow(cryptoParams *CryptoParams, matrix [][]float64) (CipherMatrix, int, int, error) {
	if len(matrix) == 0 {
		return nil, 0, 0, nil
	}
	nbrRows := len(matrix)
	d := len(matrix[0])
	matrixEnc := make(CipherMatrix, 0, nbrRows)
	for _, row := range matrix {
		if d != len(row) {
			return nil, 0, 0, errors.New("this is not a matrix (expected " + strconv.Itoa(d) + " dimensions but got " + strconv.Itoa(len(row)) + ")")
		}
		rowEnc, _ := EncryptFloatVector(cryptoParams, row)
		matrixEnc = append(matrixEnc, rowEnc)
	}
	return matrixEnc, nbrRows, d, nil
}

func EncodeFloatVector(cryptoParams *CryptoParams, f []float64) (PlainVector, int) {
	return EncodeFloatVectorWithScale(cryptoParams, f, cryptoParams.Params.DefaultScale().Float64())
}

func EncodeFloatVectorWithScale(cryptoParams *CryptoParams, f []float64, scale float64) (PlainVector, int) {
	nbrMaxCoef := cryptoParams.GetSlots()
	length := len(f)
	plainArr := make(PlainVector, 0, (length+nbrMaxCoef-1)/nbrMaxCoef)
	elementsEncoded := 0

	for elementsEncoded < length {
		start := elementsEncoded
		end := Min(elementsEncoded+nbrMaxCoef, length)
		pt := ckks.NewPlaintext(cryptoParams.Params, cryptoParams.Params.MaxLevel())
		pt.Scale = rlwe.NewScale(scale)
		if err := cryptoParams.WithEncoder(func(encoder *ckks.Encoder) error {
			return encoder.Encode(f[start:end], pt)
		}); err != nil {
			panic(err)
		}
		plainArr = append(plainArr, pt)
		elementsEncoded += end - start
	}

	return plainArr, elementsEncoded
}

func DecryptMultipleFloat(cryptoParams *CryptoParams, cipher *rlwe.Ciphertext, nbrEl int) []float64 {
	var plaintext *rlwe.Plaintext
	if err := cryptoParams.WithDecryptor(func(decryptor *rlwe.Decryptor) error {
		plaintext = decryptor.DecryptNew(cipher)
		return nil
	}); err != nil {
		panic(err)
	}

	dataDecrypted := DecodeFloatVector(cryptoParams, PlainVector{plaintext})
	if nbrEl <= 0 {
		return dataDecrypted
	}
	return dataDecrypted[:nbrEl]
}

func DecryptFloatVector(cryptoParams *CryptoParams, fEnc CipherVector, n int) []float64 {
	dataDecrypted := make([]float64, 0, Max(n, len(fEnc)*cryptoParams.GetSlots()))
	for _, cipher := range fEnc {
		dataDecrypted = append(dataDecrypted, DecryptMultipleFloat(cryptoParams, cipher, 0)...)
	}
	if n <= 0 {
		return dataDecrypted
	}
	return dataDecrypted[:n]
}

func DecryptFloatMatrix(cryptoParams *CryptoParams, matrixEnc []CipherVector, d int) [][]float64 {
	matrix := make([][]float64, 0, len(matrixEnc))
	for _, rowEnc := range matrixEnc {
		matrix = append(matrix, DecryptFloatVector(cryptoParams, rowEnc, d))
	}
	return matrix
}

func DecodeFloatVector(cryptoParams *CryptoParams, fEncoded PlainVector) []float64 {
	dataDecoded := make([]float64, 0, len(fEncoded)*cryptoParams.GetSlots())
	for _, plaintext := range fEncoded {
		values := make([]float64, cryptoParams.GetSlots())
		if err := cryptoParams.WithEncoder(func(encoder *ckks.Encoder) error {
			return encoder.Decode(plaintext, values)
		}); err != nil {
			panic(err)
		}
		dataDecoded = append(dataDecoded, values...)
	}
	return dataDecoded
}

func (cm *CipherMatrix) MarshalBinary() ([]byte, [][]int, error) {
	b := make([]byte, 0)
	ctSizes := make([][]int, len(*cm))
	for i, v := range *cm {
		tmp, n, err := v.MarshalBinary()
		if err != nil {
			return nil, nil, err
		}
		ctSizes[i] = n
		b = append(b, tmp...)
	}
	return b, ctSizes, nil
}

func (cm *CipherMatrix) UnmarshalBinary(cryptoParams *CryptoParams, f []byte, ctSizes [][]int) error {
	*cm = make([]CipherVector, len(ctSizes))
	start := 0
	for i := range ctSizes {
		rowSize := 0
		for _, size := range ctSizes[i] {
			rowSize += size
		}
		end := start + rowSize
		cv := make(CipherVector, 0, len(ctSizes[i]))
		if err := cv.UnmarshalBinary(cryptoParams, f[start:end], ctSizes[i]); err != nil {
			return err
		}
		start = end
		(*cm)[i] = cv
	}
	return nil
}

func (cv *CipherVector) MarshalBinary() ([]byte, []int, error) {
	data := make([]byte, 0)
	ctSizes := make([]int, 0, len(*cv))
	for _, ct := range *cv {
		b, err := ct.MarshalBinary()
		if err != nil {
			return nil, nil, err
		}
		data = append(data, b...)
		ctSizes = append(ctSizes, len(b))
	}
	return data, ctSizes, nil
}

func (cv *CipherVector) UnmarshalBinary(cryptoParams *CryptoParams, f []byte, fSizes []int) error {
	*cv = make(CipherVector, len(fSizes))
	start := 0
	for i, size := range fSizes {
		ct := ckks.NewCiphertext(cryptoParams.Params, 1, cryptoParams.Params.MaxLevel())
		if err := ct.UnmarshalBinary(f[start : start+size]); err != nil {
			return err
		}
		(*cv)[i] = ct
		start += size
	}
	return nil
}

func CopyEncryptedVector(src CipherVector) CipherVector {
	dest := make(CipherVector, len(src))
	for i := range src {
		if src[i] != nil {
			dest[i] = src[i].CopyNew()
		}
	}
	return dest
}

func CopyEncryptedMatrix(src []CipherVector) []CipherVector {
	dest := make([]CipherVector, len(src))
	for i := range src {
		dest[i] = CopyEncryptedVector(src[i])
	}
	return dest
}

func (cv *CipherVector) DummyBootstrapping(_ string, cryptoParams *CryptoParams) CipherVector {
	for i, ct := range *cv {
		decryptedValues := DecryptMultipleFloat(cryptoParams, ct, 0)
		encryptedValues, _ := EncryptFloatVector(cryptoParams, decryptedValues)
		(*cv)[i] = encryptedValues[0]
	}
	return *cv
}

var _ encoding.BinaryMarshaler = new(CryptoParams)
var _ encoding.BinaryUnmarshaler = new(CryptoParams)

func (cp *CryptoParams) MarshalBinary() ([]byte, error) {
	paramsBytes, err := cp.Params.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal params: %w", err)
	}

	skBytes, err := cp.Sk.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal secret key: %w", err)
	}
	aggSkBytes, err := cp.AggregateSk.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal aggregate secret key: %w", err)
	}
	pkBytes, err := cp.Pk.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal public key: %w", err)
	}

	var rlkBytes []byte
	if cp.Rlk != nil {
		evk := rlwe.NewMemEvaluationKeySet(cp.Rlk)
		rlkBytes, err = evk.MarshalBinary()
		if err != nil {
			return nil, fmt.Errorf("marshal relin key: %w", err)
		}
	}

	rotBytes := make([][]byte, len(cp.RotKs))
	for i, gk := range cp.RotKs {
		rotBytes[i], err = gk.MarshalBinary()
		if err != nil {
			return nil, fmt.Errorf("marshal galois key %d: %w", i, err)
		}
	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(cryptoParamsMarshalable{
		Params:      paramsBytes,
		Sk:          skBytes,
		AggregateSk: aggSkBytes,
		Pk:          pkBytes,
		Rlk:         rlkBytes,
		RotKs:       rotBytes,
		NumThreads:  cp.numThreads,
		Prec:        cp.prec,
	}); err != nil {
		return nil, fmt.Errorf("encode crypto params: %w", err)
	}

	return buf.Bytes(), nil
}

func (cp *CryptoParams) UnmarshalBinary(data []byte) error {
	dec := gob.NewDecoder(bytes.NewBuffer(data))
	var enc cryptoParamsMarshalable
	if err := dec.Decode(&enc); err != nil {
		return fmt.Errorf("decode crypto params: %w", err)
	}

	if err := cp.Params.UnmarshalBinary(enc.Params); err != nil {
		return fmt.Errorf("unmarshal params: %w", err)
	}

	cp.Sk = rlwe.NewSecretKey(cp.Params)
	if err := cp.Sk.UnmarshalBinary(enc.Sk); err != nil {
		return fmt.Errorf("unmarshal secret key: %w", err)
	}

	cp.AggregateSk = rlwe.NewSecretKey(cp.Params)
	if err := cp.AggregateSk.UnmarshalBinary(enc.AggregateSk); err != nil {
		return fmt.Errorf("unmarshal aggregate secret key: %w", err)
	}

	cp.Pk = rlwe.NewPublicKey(cp.Params)
	if err := cp.Pk.UnmarshalBinary(enc.Pk); err != nil {
		return fmt.Errorf("unmarshal public key: %w", err)
	}

	if len(enc.Rlk) > 0 {
		evk := &rlwe.MemEvaluationKeySet{}
		if err := evk.UnmarshalBinary(enc.Rlk); err != nil {
			return fmt.Errorf("unmarshal relin key set: %w", err)
		}
		var err error
		cp.Rlk, err = evk.GetRelinearizationKey()
		if err != nil {
			return fmt.Errorf("extract relin key: %w", err)
		}
	}

	cp.RotKs = make([]*rlwe.GaloisKey, len(enc.RotKs))
	for i, gkBytes := range enc.RotKs {
		gk := rlwe.NewGaloisKey(cp.Params)
		if err := gk.UnmarshalBinary(gkBytes); err != nil {
			return fmt.Errorf("unmarshal galois key %d: %w", i, err)
		}
		cp.RotKs[i] = gk
	}

	cp.numThreads = enc.NumThreads
	if cp.numThreads <= 0 {
		cp.numThreads = runtime.GOMAXPROCS(0)
	}
	cp.prec = enc.Prec
	cp.rebuildPools()
	return nil
}
