package cap

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"math"
)

const hashLabel = "HLL|"

// Bucket maps a variant key to its bucket, the top LogMu bits of HMAC-SHA256.
func Bucket(p Params, hashKey []byte, variant string) int {
	mac := hmac.New(sha256.New, hashKey)
	mac.Write([]byte(hashLabel))
	mac.Write([]byte(variant))
	return int(binary.BigEndian.Uint64(mac.Sum(nil)[:8]) >> (64 - p.LogMu))
}

// LocalBitmap returns the μ bucket bits of one party: 1 where at least one key lands.
func LocalBitmap(p Params, hashKey []byte, variants []string) []uint8 {
	bitmap := make([]uint8, p.Mu())
	for _, variant := range variants {
		bitmap[Bucket(p, hashKey, variant)] = 1
	}
	return bitmap
}

// EmptyBuckets returns V = #{ζ : β_ζ = 0}.
func EmptyBuckets(bitmap []uint8) int64 {
	var empty int64
	for _, bit := range bitmap {
		if bit == 0 {
			empty++
		}
	}
	return empty
}

// SelectIndex applies the threshold rule to a plain V; SecureCaps computes the same on shares.
func SelectIndex(th Thresholds, empty int64) int {
	idx := 0
	for _, tv := range th.TV {
		if empty <= tv {
			idx++
		}
	}
	return idx
}

// Thresholds is the public threshold table; it depends only on Params.
type Thresholds struct {
	Ladder []int   // L_0..L_J
	TV     []int64 // rung i < J: m̂ = μ·ln(μ/V) > L′_i = L_i/(1+ε) ⟺ V ≤ TV[i]
}

// BuildThresholds sets TV[i] = ⌈μ·e^(−L′_i/μ)⌉ − 1, so that idx = #{i : V ≤ TV[i]} is the index of
// the smallest rung L with L ≥ (1+ε)·m̂ (capped at J). V = 0 gives idx = J.
func BuildThresholds(p Params) Thresholds {
	mu := float64(p.Mu())
	ladder := p.Ladder()
	th := Thresholds{Ladder: ladder, TV: make([]int64, p.J)}
	for i := range th.TV {
		target := float64(ladder[i]) / (1 + p.Epsilon)
		th.TV[i] = int64(math.Ceil(mu*math.Exp(-target/mu))) - 1
	}
	return th
}
