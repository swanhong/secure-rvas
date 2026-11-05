package embed

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/cmplx"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Encode maps the 32 base-four digits of a fingerprint to i^a, low digit first.
// The zero fingerprint is a real code (all ones); a dummy code is [32]complex128{}.
func Encode(fingerprint uint64) [32]complex128 {
	roots := [4]complex128{1, 1i, -1, -1i}
	var code [32]complex128
	for d := range code {
		code[d] = roots[(fingerprint>>uint(2*d))&3]
	}
	return code
}

// EncodeKey uses the agreed first 64 SHA-256 bits, read in big-endian order.
// The empty string is a real key. Dummy rows must explicitly use the zero code.
func EncodeKey(key string) [32]complex128 {
	digest := sha256.Sum256([]byte(key))
	return Encode(binary.BigEndian.Uint64(digest[:8]))
}

// encodeLocalKeys caches all 32 planes once per local key for a table write.
func encodeLocalKeys(keys [][]string) [][][32]complex128 {
	codes := make([][][32]complex128, len(keys))
	for g, local := range keys {
		codes[g] = make([][32]complex128, len(local))
		for j, key := range local {
			codes[g][j] = EncodeKey(key)
		}
	}
	return codes
}

func conjugate(cp *crypto.CryptoParams, x *rlwe.Ciphertext) *rlwe.Ciphertext {
	var out *rlwe.Ciphertext
	if err := cp.WithEvaluator(func(eval *ckks.Evaluator) error {
		var err error
		out, err = eval.ConjugateNew(x)
		return err
	}); err != nil {
		panic(err)
	}
	return out
}

// EqualityInput is one full ciphertext's query-cell pairs, already placed in
// corresponding slots. Only the evaluating site supplies it. Query[d][s] is a
// local plaintext constant; F[d] is a ciphertext under the collective key.
// Dummy queries and empty cells use zero codes, which give distance h=32.
// This component neither routes pairs nor selects/skips candidate cells.
type EqualityInput struct {
	Query [32][]complex128
	F     [32]*rlwe.Ciphertext
}

type EqualityStep struct {
	Name     string
	Level    int
	LogScale float64
}

type EqualityResult struct {
	// Match is the polynomial result, without occupancy or real-row gates.
	// Match and Steps are returned only at the evaluating site.
	Match     *rlwe.Ciphertext
	Steps     []EqualityStep
	Refreshes int
}

// Equal evaluates key equality without decryption or separate gates.

func Equal(cp *crypto.CryptoParams, net *mpc.Network, source, inputLevel int, in EqualityInput) EqualityResult {
	return equalWithRefresh(cp, net, source, inputLevel, in, nil)
}

// Hash sources supply a refresh callback while peers service requests.
func equalWithRefresh(cp *crypto.CryptoParams, net *mpc.Network, source, inputLevel int, in EqualityInput, refresh func(*rlwe.Ciphertext) *rlwe.Ciphertext) EqualityResult {
	if source < 1 || source >= net.GetNParty() {
		panic("Equal: source must be a data site")
	}
	if inputLevel > cp.Params.MaxLevel() || !net.CanCollectiveBootstrap(cp, inputLevel-2) {
		panic("Equal: input level must reserve two levels above the refresh minimum")
	}
	result := EqualityResult{}
	if net.GetPid() == 0 {
		return result
	}
	owner := net.GetPid() == source
	if owner {
		for d, ct := range in.F {
			if len(in.Query[d]) != cp.GetSlots() || ct == nil || ct.Level() != inputLevel || ct.Degree() != 1 || !ct.IsBatched || ct.Slots() != cp.GetSlots() {
				panic("Equal: key plane/query shape or level mismatch")
			}
			if ct.Scale.Cmp(in.F[0].Scale) != 0 {
				panic("Equal: key plane scales differ")
			}
		}
	}
	var v *rlwe.Ciphertext
	level := inputLevel
	record := func(name string) {
		if owner {
			if v.Level() != level {
				panic("Equal: public level schedule disagrees with ciphertext")
			}
			result.Steps = append(result.Steps, EqualityStep{name, level, math.Log2(v.Scale.Float64())})
		}
	}
	reserve := func(depth int) {
		if !net.CanCollectiveBootstrap(cp, level-depth) {
			if refresh != nil {
				v = refresh(v)
			} else {
				var live crypto.CipherMatrix
				if owner {
					live = crypto.CipherMatrix{{v}}
				}
				live = net.RefreshCMatWithRandomZero(cp, live, source, 1, 1)
				v = live[0][0]
			}
			level = cp.Params.MaxLevel()
			result.Refreshes++
			record("refresh")
		}
	}
	mul := func(a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
		return crypto.CMult(cp, crypto.CipherVector{a}, crypto.CipherVector{b})[0]
	}
	if owner {
		for d, ct := range in.F {
			coefficients := make([]complex128, cp.GetSlots())
			for s, x := range in.Query[d] {
				coefficients[s] = (2.0 / 81) * cmplx.Conj(x)
			}
			pt, _ := crypto.EncodeComplexVector(cp, coefficients)
			term := crypto.CPMult(cp, crypto.CipherVector{ct}, pt)[0]
			v = addCiphertext(cp, v, term)
		}
		v = crypto.AddConst(cp, crypto.Add(cp, v, conjugate(cp, v)), 2.0/81)
	}
	level--
	record("v0")
	for k := 0; k < 6; k++ {
		reserve(1)
		if owner {
			v = crypto.AddConst(cp, mul(v, v), -2*math.Pow(7.0/9, math.Ldexp(1, k+1)))
		}
		level--
		record(fmt.Sprintf("v%d", k+1))
	}
	reserve(2) // Keep both v^2 and 3-2v live within this block.
	if owner {
		square := mul(v, v)
		linear := crypto.CMultConst(cp, crypto.CipherVector{v}, -2, false)[0]
		linear = crypto.AddConst(cp, linear, 3)
		v = mul(square, linear)
		result.Match = v
	}
	level -= 2
	record("cleanup")
	return result
}
