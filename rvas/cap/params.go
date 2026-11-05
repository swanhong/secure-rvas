package cap

// Params are the public parameters of the cap step. Each party sketches a gene's keys as a
// linear-counting bitmap of μ buckets (equivalent to keeping only the zero/nonzero status of
// HyperLogLog registers), the estimate is m̂ = μ·ln(μ/V) with V empty buckets, and the cap is the
// smallest ladder rung holding (1+ε)·m̂.
type Params struct {
	LogMu   int     `toml:"log_mu"`  // b, number of buckets μ = 2^b
	Epsilon float64 `toml:"epsilon"` // allocation margin ε
	Floor   int     `toml:"floor"`   // smallest cap L_0
	J       int     `toml:"j"`       // ladder L_i = Floor·2^i for i = 0..J, N_max = L_J
}

// DefaultParams uses μ = 2048 and the margin ε = 0.1625.
func DefaultParams() Params {
	return Params{LogMu: 11, Epsilon: 0.1625, Floor: 4, J: 11}
}

func (p Params) Mu() int { return 1 << p.LogMu }

// returns L_0..L_J.
func (p Params) Ladder() []int {
	ladder := make([]int, p.J+1)
	for i := range ladder {
		ladder[i] = p.Floor << i
	}
	return ladder
}

func (p Params) NMax() int { return p.Floor << p.J }
