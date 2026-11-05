package mpc

import (
	crand "crypto/rand"
	"fmt"
	"math"
	"math/big"

	mpc_core "github.com/hhcho/mpc-core"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

const (
	conversionLambda       = 128
	conversionHeadroomBits = 8
	conversionNoise        = 19
)

func ceilLog2Int(x *big.Int) int {
	if x.Sign() <= 0 {
		return 0
	}
	return new(big.Int).Sub(x, big.NewInt(1)).BitLen()
}

func checkSSToCKKSHeadroom(params ckks.Parameters, rtype mpc_core.RElem, fracBits, nParty int, scale rlwe.Scale, level int) error {
	dataParties := max(1, nParty-1)
	encodedShares := new(big.Int).Rsh(rtype.Modulus(), 1)
	encodedShares.Mul(encodedShares, scale.BigInt())
	encodedShares.Mul(encodedShares, big.NewInt(int64(dataParties)))
	encodedShares.Lsh(encodedShares, conversionHeadroomBits)

	qHalf := new(big.Int).Rsh(new(big.Int).Set(params.RingQ().ModulusAtLevel[level]), 1)
	fieldScale := new(big.Int).Lsh(big.NewInt(1), uint(fracBits))
	if encodedShares.Cmp(new(big.Int).Mul(qHalf, fieldScale)) >= 0 {
		required := ceilLog2Int(encodedShares) - fracBits
		available := qHalf.BitLen() - 1
		return fmt.Errorf("SS->CKKS: level=%d required_bits=%d available_bits=%d", level, required, available)
	}
	return nil
}

func checkCMatToSSAdmission(params ckks.Parameters, cm crypto.CipherMatrix, level, nParty, log2MaxMagnitude int) error {
	dataParties := max(1, nParty-1)
	q := params.RingQ().ModulusAtLevel[level]
	maskRange := new(big.Int).Quo(new(big.Int).Set(q), big.NewInt(int64(2*dataParties)))
	maskSupport := new(big.Int).Rsh(new(big.Int).Add(maskRange, big.NewInt(1)), 1)

	scaleBits := 0
	for i := range cm {
		for j := range cm[i] {
			scaleBits = max(scaleBits, int(math.Ceil(cm[i][j].Scale.Log2())))
		}
	}
	messageBits := max(0, scaleBits+log2MaxMagnitude)
	// Preserve the statistical target across all coefficients in one ciphertext.
	required := messageBits + conversionLambda + params.LogN()
	available := maskSupport.BitLen() - 1
	if required > available {
		return fmt.Errorf("CMatToSS: level=%d required_bits=%d available_bits=%d", level, required, available)
	}

	// Centered party masks occupy at most Q/4; keep the bounded message and
	// conversion noise inside the remaining centered-Q headroom.
	total := new(big.Int).Mul(maskSupport, big.NewInt(int64(dataParties)))
	total.Add(total, new(big.Int).Lsh(big.NewInt(1), uint(messageBits)))
	total.Add(total, big.NewInt(int64(conversionNoise*dataParties)))
	qHalf := new(big.Int).Rsh(new(big.Int).Set(q), 1)
	if total.Cmp(qHalf) >= 0 {
		return fmt.Errorf("CMatToSS: level=%d required_bits=%d available_bits=%d", level, ceilLog2Int(total), qHalf.BitLen()-1)
	}
	return nil
}

func plaintextFromPoly(cryptoParams *crypto.CryptoParams, poly ring.Poly, src *rlwe.Ciphertext, level int) *rlwe.Plaintext {
	pt := ckks.NewPlaintext(cryptoParams.Params, level)
	pt.Value = poly
	pt.Element.Value[0] = poly
	pt.Scale = src.Scale
	pt.IsNTT = src.IsNTT
	return pt
}

func (mpcObj *MPC) SSMultMat(a, b mpc_core.RMat) mpc_core.RMat {
	ar, am := mpcObj.BeaverPartitionMat(a)
	br, bm := mpcObj.BeaverPartitionMat(b)
	ab := mpcObj.BeaverMultMat(ar, am, br, bm)
	return mpcObj.BeaverReconstructMat(ab)
}

// SSMultMats returns a[i]·b[i] for every pair with the masked openings of all left and all right
// operands done together (one per side) and one reconstruction; the pairs may differ in shape.
// Result rows are capacity-limited views of one shared array.
func (mpcObj *MPC) SSMultMats(a, b []mpc_core.RMat) []mpc_core.RMat {
	flatten := func(matrices []mpc_core.RMat) mpc_core.RMat {
		var flat mpc_core.RVec
		for _, matrix := range matrices {
			for _, row := range matrix {
				flat = append(flat, row...)
			}
		}
		return mpc_core.RMat{flat}
	}
	reshape := func(flat mpc_core.RVec, start, rows, cols int) mpc_core.RMat {
		matrix := make(mpc_core.RMat, rows)
		for r := range matrix {
			matrix[r] = flat[start+r*cols : start+(r+1)*cols : start+(r+1)*cols]
		}
		return matrix
	}

	ar, am := mpcObj.BeaverPartitionMat(flatten(a))
	br, bm := mpcObj.BeaverPartitionMat(flatten(b))
	products := make([]mpc_core.RMat, len(a))
	startA, startB := 0, 0
	for i := range a {
		rowsA, colsA := a[i].Dims()
		rowsB, colsB := b[i].Dims()
		products[i] = mpcObj.BeaverMultMat(
			reshape(ar[0], startA, rowsA, colsA), reshape(am[0], startA, rowsA, colsA),
			reshape(br[0], startB, rowsB, colsB), reshape(bm[0], startB, rowsB, colsB),
		)
		startA += rowsA * colsA
		startB += rowsB * colsB
	}

	flat := mpcObj.BeaverReconstructMat(flatten(products))[0]
	start := 0
	for i := range products {
		rows, cols := products[i].Dims()
		products[i] = reshape(flat, start, rows, cols)
		start += rows * cols
	}
	return products
}

func (mpcObj *MPC) SSMultElemVecScalar(a mpc_core.RVec, b mpc_core.RElem) mpc_core.RVec {
	ar, am := mpcObj.BeaverPartitionVec(a)
	br, bm := mpcObj.BeaverPartition(b)
	x := mpc_core.InitRVec(mpcObj.rtype.Zero(), len(a))
	for i := range x {
		x[i] = mpcObj.BeaverMult(ar[i], am[i], br, bm)
	}
	return mpcObj.BeaverReconstructVec(x)
}

func (mpcObj *MPC) SSSquareElemVec(a mpc_core.RVec) mpc_core.RVec {
	ar, am := mpcObj.BeaverPartitionVec(a)
	x := mpcObj.BeaverMultElemVec(ar, am, ar, am)
	return mpcObj.BeaverReconstructVec(x)
}

func (mpcObj *MPC) SSMultElemVec(a, b mpc_core.RVec) mpc_core.RVec {
	ar, am := mpcObj.BeaverPartitionVec(a)
	br, bm := mpcObj.BeaverPartitionVec(b)
	x := mpcObj.BeaverMultElemVec(ar, am, br, bm)
	return mpcObj.BeaverReconstructVec(x)
}

func (mpcObj *MPC) SSMultElemMat(a, b mpc_core.RMat) mpc_core.RMat {
	ar, am := mpcObj.BeaverPartitionMat(a)
	br, bm := mpcObj.BeaverPartitionMat(b)
	x := mpcObj.BeaverMultElemMat(ar, am, br, bm)
	return mpcObj.BeaverReconstructMat(x)
}

// --- Secure share <-> ciphertext conversion (slot-domain, masked) ---
//
// Re-port of the v2 masked conversions (which used the patched Lattigo-v2 encoder methods
// EncodeRVecNew/DecodeRVec) to stock Lattigo v6, via full-precision []*big.Float slot
// encoding so field-sized masks survive.

// elemToSignedBigFloat returns the centered (signed) fixed-point real value of e.
// Only the field types used by this codebase (LElem128/LElem256) are supported.
func elemToSignedBigFloat(e mpc_core.RElem, fracBits int) *big.Float {
	switch v := e.(type) {
	case mpc_core.LElem256:
		return v.ToSignedBigFloat(fracBits)
	case mpc_core.LElem128:
		return v.ToSignedBigFloat(fracBits)
	default:
		panic("mpc: SS<->cipher conversion only supports LElem128 or LElem256")
	}
}

// signedBigFloatToElem converts a signed fixed-point real value back into a field
// element: round(f * 2^fracBits) reduced modulo the field modulus. The explicit
// reduction is required because RElem.FromBigInt does not reduce on construction.
func signedBigFloatToElem(rtype mpc_core.RElem, f *big.Float, fracBits int) mpc_core.RElem {
	prec := f.Prec()
	if prec < 256 {
		prec = 256
	}
	scaled := new(big.Float).SetPrec(prec).Mul(f, bigFloatPow2(fracBits, prec))
	half := new(big.Float).SetPrec(prec).SetFloat64(0.5)
	if scaled.Sign() >= 0 {
		scaled.Add(scaled, half)
	} else {
		scaled.Sub(scaled, half)
	}
	r, _ := scaled.Int(nil)
	r.Mod(r, rtype.Modulus())
	return rtype.FromBigInt(r)
}

func bigFloatPow2(n int, prec uint) *big.Float {
	return new(big.Float).SetPrec(prec).SetInt(new(big.Int).Lsh(big.NewInt(1), uint(n)))
}

// sampleCenteredMask draws a per-element uniform mask in [-bound/2, bound/2), stored
// in the field representation. Each party samples its own mask locally.
func sampleCenteredMask(rtype mpc_core.RElem, nrow, ncol int, bound *big.Int) mpc_core.RMat {
	modulus := rtype.Modulus()
	boundHalf := new(big.Int).Rsh(bound, 1)
	mask := make(mpc_core.RMat, nrow)
	for i := range mask {
		mask[i] = make(mpc_core.RVec, ncol)
		for j := range mask[i] {
			t, err := crand.Int(crand.Reader, bound)
			if err != nil {
				panic(err)
			}
			if t.Cmp(boundHalf) >= 0 {
				t.Sub(t, bound)
			}
			t.Mod(t, modulus)
			mask[i][j] = rtype.FromBigInt(t)
		}
	}
	return mask
}

// conversionEncoder retains the low-order signal while encoding or decoding
// field-sized masks. A 53-bit encoder would convert even big.Float inputs to
// float64. Cache by full parameters because the FFT roots depend on the ring.
func (mpcObj *MPC) conversionEncoder(params ckks.Parameters) *ckks.Encoder {
	if mpcObj.hpEncoder == nil || !mpcObj.hpEncoder.GetParameters().Equal(&params) {
		mpcObj.hpEncoder = ckks.NewEncoder(params, uint(params.LogQP()+128))
	}
	return mpcObj.hpEncoder
}

// encodeElemSlotsToCM encrypts row-major slot-domain field shares into a CipherMatrix
// under the collective public key, using high-precision arithmetic for masked values.
// scale/level set the encoding of the resulting ciphertexts.
func (mpcObj *MPC) encodeElemSlotsToCM(cryptoParams *crypto.CryptoParams, share mpc_core.RMat, scale rlwe.Scale, level int) crypto.CipherMatrix {
	encoder := mpcObj.conversionEncoder(cryptoParams.Params)
	slots := cryptoParams.GetSlots()
	fracBits := mpcObj.GetFracBits()
	numCtxRow := len(share)
	nElemCol := len(share[0])
	numCtxCol := 1 + ((nElemCol - 1) / slots)

	pm := make(crypto.PlainMatrix, numCtxRow)
	for i := range pm {
		pm[i] = make(crypto.PlainVector, numCtxCol)
		for j := 0; j < numCtxCol; j++ {
			start := j * slots
			end := start + slots
			if end > nElemCol {
				end = nElemCol
			}
			buf := make([]*big.Float, slots)
			for k := range buf {
				buf[k] = new(big.Float)
			}
			for k := start; k < end; k++ {
				buf[k-start] = elemToSignedBigFloat(share[i][k], fracBits)
			}
			pt := ckks.NewPlaintext(cryptoParams.Params, level)
			pt.Scale = scale
			if err := encoder.Encode(buf, pt); err != nil {
				panic(err)
			}
			pm[i][j] = pt
		}
	}

	return crypto.EncryptPlaintextMatrix(cryptoParams, pm)
}

// decodePlainToElemSlots decodes plaintexts back to slot-domain field elements
// (nElemRow per row), reversing encodeElemSlotsToCM.
func (mpcObj *MPC) decodePlainToElemSlots(cryptoParams *crypto.CryptoParams, rtype mpc_core.RElem, pm crypto.PlainMatrix, nElemRow int) mpc_core.RMat {
	slots := cryptoParams.GetSlots()
	fracBits := mpcObj.GetFracBits()
	numCtxRow := len(pm)

	encoder := mpcObj.conversionEncoder(cryptoParams.Params)
	decPrec := encoder.Prec()

	out := mpc_core.InitRMat(rtype.Zero(), numCtxRow, nElemRow)
	for i := range pm {
		for j := range pm[i] {
			buf := make([]*big.Float, slots)
			for k := range buf {
				buf[k] = new(big.Float).SetPrec(decPrec)
			}
			if err := encoder.Decode(pm[i][j], buf); err != nil {
				panic(err)
			}
			start := j * slots
			for k := 0; k < slots && start+k < nElemRow; k++ {
				out[i][start+k] = signedBigFloatToElem(rtype, buf[k], fracBits)
			}
		}
	}
	return out
}

// SSToCMat converts secret-shared values (row-major, slot-domain) into a CipherMatrix
// encrypting the reconstructed value. Faithful to the v2 masked-share protocol: each
// party masks its share, only x - Sum(mask_i) is revealed, and per-party encryptions of
// the masked share are summed to Enc(x).
func (mpcObj *MPC) SSToCMat(cryptoParams *crypto.CryptoParams, rm mpc_core.RMat) (cm crypto.CipherMatrix) {
	if err := checkSSToCKKSHeadroom(cryptoParams.Params, mpcObj.GetRType(), mpcObj.GetFracBits(), mpcObj.GetNParty(), cryptoParams.Params.DefaultScale(), cryptoParams.Params.MaxLevel()); err != nil {
		panic(err)
	}
	if mpcObj.GetPid() == 0 {
		cm = make(crypto.CipherMatrix, 1)
		cm[0] = make(crypto.CipherVector, 1)
		return
	}

	rtype := rm.Type().Zero()
	if rtype.TypeID() != mpc_core.LElem256UniqueID && rtype.TypeID() != mpc_core.LElem128UniqueID {
		panic("SSToCMat only supported for LElem128 or LElem256")
	}

	numCtxRow := len(rm)
	nElemCol := len(rm[0])

	// Mask bound: modulus / (4 * #data-parties), as in v2.
	bound := new(big.Int).Set(rtype.Modulus())
	bound.Quo(bound, big.NewInt(int64(4*(mpcObj.GetNParty()-1))))

	mask := sampleCenteredMask(rtype, numCtxRow, nElemCol, bound)

	rmMask := rm.Copy()
	rmMask.Sub(mask)
	rmMask = mpcObj.RevealSymMat(rmMask) // reveals only x - Sum(mask_i)

	var share mpc_core.RMat
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		share = rmMask
		share.Add(mask)
	} else {
		share = mask
	}

	cm = mpcObj.encodeElemSlotsToCM(cryptoParams, share, cryptoParams.Params.DefaultScale(), cryptoParams.Params.MaxLevel())
	return mpcObj.Network.AggregateCMat(cryptoParams, cm)
}

func (mpcObj *MPC) SSToCVec(cryptoParams *crypto.CryptoParams, rv mpc_core.RVec) (cv crypto.CipherVector) {
	return mpcObj.SSToCMat(cryptoParams, mpc_core.RMat{rv})[0]
}

func (mpcObj *MPC) SStoCiphertext(cryptoParams *crypto.CryptoParams, rv mpc_core.RVec) *rlwe.Ciphertext {
	return mpcObj.SSToCVec(cryptoParams, rv)[0]
}

func (mpcObj *MPC) CMatToSS(cryptoParams *crypto.CryptoParams, rtype mpc_core.RElem, cm crypto.CipherMatrix, sourcePid, numCtxRow, numCtxCol, nElemRow int, maxMagnitudeBits ...int) (rm mpc_core.RMat) {
	rm = mpc_core.InitRMat(rtype.Zero(), numCtxRow, nElemRow)
	if mpcObj.GetPid() == 0 {
		return
	}
	if rtype.TypeID() != mpc_core.LElem256UniqueID && rtype.TypeID() != mpc_core.LElem128UniqueID {
		panic("CMatToSS only supported for LElem128 or LElem256")
	}

	if sourcePid > 0 {
		cm = mpcObj.Network.BroadcastCMatWithRandomZero(cryptoParams, cm, sourcePid, numCtxRow, numCtxCol)
	}

	cm, level := crypto.FlattenLevels(cryptoParams, cm)
	magnitudeBits := mpcObj.GetDataBits() - mpcObj.GetFracBits()
	if len(maxMagnitudeBits) > 0 {
		magnitudeBits = maxMagnitudeBits[0]
	}
	if err := checkCMatToSSAdmission(cryptoParams.Params, cm, level, mpcObj.GetNParty(), magnitudeBits); err != nil {
		panic(err)
	}

	ringQ := cryptoParams.Params.RingQ().AtLevel(level)
	nCoeffs := cryptoParams.Params.N()
	nParty := mpcObj.GetNParty()

	// v2 mask bound = (product of active Q moduli) / (2 * #data-parties).
	bound := new(big.Int).Set(cryptoParams.Params.RingQ().ModulusAtLevel[level])
	bound.Quo(bound, big.NewInt(int64(2*(nParty-1))))
	boundHalf := new(big.Int).Rsh(bound, 1)

	// Private smudge sampler (independent PRNG, never the shared CRS).
	prng, err := sampling.NewPRNG()
	if err != nil {
		panic(err)
	}
	gauss := ring.NewGaussianSampler(prng, cryptoParams.Params.RingQ(), ring.DiscreteGaussian{Sigma: 3.19, Bound: 19}, false).AtLevel(level)

	// Per ciphertext, build the masked partial-decryption share h0 = sk*c1 + mask + e
	// (coefficient-domain mask, NTT'd), keeping each mask poly locally for the decode.
	maskPolys := make([][]ring.Poly, numCtxRow)
	shares := make([][]ring.Poly, numCtxRow)
	for i := range cm {
		maskPolys[i] = make([]ring.Poly, len(cm[i]))
		shares[i] = make([]ring.Poly, len(cm[i]))
		for j := range cm[i] {
			maskBig := make([]*big.Int, nCoeffs)
			for t := range maskBig {
				m, cerr := crand.Int(crand.Reader, bound)
				if cerr != nil {
					panic(cerr)
				}
				if m.Cmp(boundHalf) >= 0 {
					m.Sub(m, bound)
				}
				maskBig[t] = m
			}
			maskPoly := ringQ.NewPoly()
			ringQ.SetCoefficientsBigint(maskBig, maskPoly)
			ringQ.NTT(maskPoly, maskPoly)
			maskPolys[i][j] = maskPoly

			h0 := *maskPoly.CopyNew()
			ringQ.MulCoeffsMontgomeryThenAdd(cryptoParams.Sk.Value.Q, cm[i][j].Value[1], h0)
			eNoise := gauss.ReadNew()
			ringQ.NTT(eNoise, eNoise)
			ringQ.Add(h0, eNoise, h0)
			shares[i][j] = h0
		}
	}

	// Aggregate the decryption shares across parties: agg = sk*c1 + Sum mask + Sum e.
	agg := make([][]ring.Poly, numCtxRow)
	for i := range shares {
		agg[i] = mpcObj.Network.AggregateDecryptShareVec(cryptoParams, shares[i], level)
	}

	// c0 + agg encodes the masked plaintext (x + Sum mask); decode it and the local mask
	// to the field (linear big.Float slot decode = v2's patched DecodeRVec).
	slots := cryptoParams.GetSlots()
	maskWidth := len(cm[0]) * slots
	pmSum := make(crypto.PlainMatrix, numCtxRow)
	pmMask := make(crypto.PlainMatrix, numCtxRow)
	for i := range cm {
		pmSum[i] = make(crypto.PlainVector, len(cm[i]))
		pmMask[i] = make(crypto.PlainVector, len(cm[i]))
		for j := range cm[i] {
			sumPoly := *cm[i][j].Value[0].CopyNew()
			ringQ.Add(sumPoly, agg[i][j], sumPoly)
			pmSum[i][j] = plaintextFromPoly(cryptoParams, sumPoly, cm[i][j], level)
			pmMask[i][j] = plaintextFromPoly(cryptoParams, maskPolys[i][j], cm[i][j], level)
		}
	}

	decodedMask := mpcObj.decodePlainToElemSlots(cryptoParams, rtype, pmMask, maskWidth)

	// v2 sign convention: hub = decode(x + Sum mask) - mask_hub ; others = -mask_i.
	var shareFull mpc_core.RMat
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		shareFull = mpcObj.decodePlainToElemSlots(cryptoParams, rtype, pmSum, maskWidth)
		shareFull.Sub(decodedMask)
	} else {
		shareFull = mpc_core.InitRMat(rtype.Zero(), numCtxRow, maskWidth)
		shareFull.Sub(decodedMask)
	}

	// Output only the nElemRow data slots (element k lives at flat index k).
	for i := range rm {
		for j := 0; j < nElemRow; j++ {
			rm[i][j] = shareFull[i][j]
		}
	}
	return
}

func (mpcObj *MPC) CVecToSS(cryptoParams *crypto.CryptoParams, rtype mpc_core.RElem, cv crypto.CipherVector, sourcePid, numCtx, nElem int, maxMagnitudeBits ...int) (rm mpc_core.RVec) {
	return mpcObj.CMatToSS(cryptoParams, rtype, crypto.CipherMatrix{cv}, sourcePid, 1, numCtx, nElem, maxMagnitudeBits...)[0]
}

func (mpcObj *MPC) CiphertextToSS(cryptoParams *crypto.CryptoParams, rtype mpc_core.RElem, ct *rlwe.Ciphertext, sourcePid, n int, maxMagnitudeBits ...int) (rv mpc_core.RVec) {
	return mpcObj.CVecToSS(cryptoParams, rtype, crypto.CipherVector{ct}, sourcePid, 1, n, maxMagnitudeBits...)
}
