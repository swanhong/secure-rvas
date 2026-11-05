package embed

import (
	"fmt"
	"math/big"
)

// HashPolicy fixes the total table size and number of proposal stages.
// Zero fields use R=2M, H=7 and the uniform-address model target 2^-41.
type HashPolicy struct {
	Expansion   int `toml:"expansion"`
	H           int `toml:"h"`
	FailureBits int `toml:"failure_bits"`
}

func (p HashPolicy) defaults() HashPolicy {
	if p.Expansion == 0 {
		p.Expansion = 2
	}
	if p.H == 0 {
		p.H = 7
	}
	if p.FailureBits == 0 {
		p.FailureBits = 41
	}
	return p
}

// TableSetting uses dense for M <= denseMax in auto mode, hash otherwise.
func TableSetting(mode string, m, slots int, policy HashPolicy, denseMax int) (Setting, error) {
	if m < 1 || m&(m-1) != 0 || m > slots {
		return Setting{}, fmt.Errorf("embedding: cap %d must be a power of two fitting %d slots", m, slots)
	}
	p := policy.defaults()
	if mode == "hash" || mode == "auto" {
		if p.FailureBits < 1 || p.FailureBits > 256 {
			return Setting{}, fmt.Errorf("embedding: hash failure_bits must be in [1,256]")
		}
		if p.Expansion < 2 || p.Expansion&(p.Expansion-1) != 0 || p.H < 1 {
			return Setting{}, fmt.Errorf("embedding: hash needs power-of-two expansion >= 2 and H > 0")
		}
	}
	if mode == "auto" {
		if denseMax < 1 || denseMax&(denseMax-1) != 0 {
			return Setting{}, fmt.Errorf("embedding: auto needs an explicit power-of-two dense_max >= 1")
		}
		mode = "hash"
		if m <= denseMax {
			mode = "dense"
		}
	}

	switch mode {
	case "dense":
		return Setting{M: m, Sigma: m}, nil
	case "hash":
		if p.Expansion > slots/m {
			return Setting{}, fmt.Errorf("embedding: hash needs expansion*M <= %d", slots)
		}
		r := p.Expansion * m
		for s := 0; s < m; s++ {
			k := int64(s + 1)
			exponent := new(big.Int).Mul(big.NewInt(int64(p.H)), big.NewInt(k))
			numerator := new(big.Int).Exp(big.NewInt(int64(m-s-1)), exponent, nil)
			numerator.Mul(numerator, new(big.Int).Binomial(int64(m), k))
			numerator.Lsh(numerator, uint(p.FailureBits))
			denominator := new(big.Int).Exp(big.NewInt(int64(r-s)), exponent, nil)
			if numerator.Cmp(denominator) <= 0 {
				return Setting{M: m, Rho: r - s, Sigma: s, H: p.H}, nil
			}
		}
	default:
		return Setting{}, fmt.Errorf("embedding: mode must be dense, hash or auto, got %q", mode)
	}
	return Setting{}, fmt.Errorf("embedding: no hash setting for cap %d", m)
}
