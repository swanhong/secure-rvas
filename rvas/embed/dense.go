package embed

import (
	"fmt"
	"math"
	"math/bits"
	"math/cmplx"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// DenseShape is public. Each gene occupies M consecutive table slots; genes
// beyond one ciphertext continue in the next chunk, using the same layout.
type DenseShape struct{ M, Genes int }

const (
	denseO = 32 + iota
	denseCount
	densePlanes
)

// newCipherMatrix allocates a rectangular container with nil ciphertext entries.
// It does not encrypt zeros or consume randomness.
func newCipherMatrix(rows, columns int) crypto.CipherMatrix {
	cm := make(crypto.CipherMatrix, rows)
	for i := range cm {
		cm[i] = make(crypto.CipherVector, columns)
	}
	return cm
}

func newPlanes(planes, slots int) [][]complex128 {
	values := make([][]complex128, planes)
	for d := range values {
		values[d] = make([]complex128, slots)
	}
	return values
}

// encryptPlanes writes one ciphertext per plane into the selected chunk.
func encryptPlanes(cp *crypto.CryptoParams, cm crypto.CipherMatrix, chunk int, values [][]complex128) {
	for d, values := range values {
		cv, _ := crypto.EncryptComplexVector(cp, values) // Second result is input length.
		cm[d][chunk] = cv[0]
	}
}

// addCiphertext starts a sum without an extra encryption of zero.
func addCiphertext(cp *crypto.CryptoParams, a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	if a == nil {
		return b
	}
	return crypto.Add(cp, a, b)
}

func (s DenseShape) chunks(cp *crypto.CryptoParams) int {
	if s.M < 1 || s.M&(s.M-1) != 0 || s.M > cp.GetSlots() || s.Genes < 0 {
		panic("dense: invalid public shape")
	}
	perChunk := cp.GetSlots() / s.M
	return (s.Genes + perChunk - 1) / perChunk
}

func (s DenseShape) genes(cp *crypto.CryptoParams, chunk int) int {
	perChunk := cp.GetSlots() / s.M
	return min(perChunk, s.Genes-chunk*perChunk)
}

type DenseTable struct {
	F        [32]crypto.CipherVector
	O, Count crypto.CipherVector
}

// Matrix orders the existing ciphertexts as F0..F31,O,Count; it does not copy.
func (t DenseTable) Matrix() crypto.CipherMatrix {
	cm := make(crypto.CipherMatrix, densePlanes)
	copy(cm, t.F[:])
	cm[denseO], cm[denseCount] = t.O, t.Count
	return cm
}

func DenseTableFromMatrix(cm crypto.CipherMatrix) DenseTable {
	if len(cm) != densePlanes {
		panic("dense: expected 32 key planes, occupancy, count")
	}
	var t DenseTable
	copy(t.F[:], cm[:32])
	t.O, t.Count = cm[denseO], cm[denseCount]
	return t
}

// BuildDenseAnchor is local to P1. It encrypts the paper's plaintext anchor
// table under the collective public key. Count is repeated within each gene's
// block; it is not reconstructed from occupancy in subsequent turns.
func BuildDenseAnchor(cp *crypto.CryptoParams, shape DenseShape, keys [][]string) DenseTable {
	chunks := shape.chunks(cp)
	if len(keys) != shape.Genes {
		panic("BuildDenseAnchor: local gene count mismatch")
	}
	cm := newCipherMatrix(densePlanes, chunks)
	for c := 0; c < chunks; c++ {
		values := newPlanes(densePlanes, cp.GetSlots())
		for g := 0; g < shape.genes(cp, c); g++ {
			local := keys[c*(cp.GetSlots()/shape.M)+g]
			if len(local) > shape.M {
				panic("BuildDenseAnchor: local rows exceed M")
			}
			for j, key := range local {
				for d, x := range EncodeKey(key) {
					values[d][g*shape.M+j] = x
				}
				values[denseO][g*shape.M+j] = 1
			}
			for j := 0; j < shape.M; j++ {
				values[denseCount][g*shape.M+j] = complex(float64(len(local)), 0)
			}
		}
		encryptPlanes(cp, cm, c, values)
	}
	return DenseTableFromMatrix(cm)
}

type TableStage struct {
	Name                     string
	Level, CiphertextBytes   int
	MinLogScale, MaxLogScale float64
}

// TableCost counts arithmetic in the source (excluding refresh internals).
// Stage byte counts describe the listed live values, not process peak memory.
type TableCost struct {
	PlainProducts, CipherProducts, Rotations, Conjugations int
	RefreshCiphertexts, RefreshBatches                     int
	Stages                                                 []TableStage
}

type tableCircuit struct {
	cp        *crypto.CryptoParams
	net       *mpc.Network
	source    int
	cost      TableCost
	requested bool // Only hash lookup/update use the refresh request loop.
}

func (c *tableCircuit) owner() bool { return c.net.GetPid() == c.source }

// equal accounts for the fixed source circuit and its collective refreshes.
// 32 key products; six squares + two cleanup products.
func (c *tableCircuit) equal(level int, in EqualityInput) EqualityResult {
	var refresh func(*rlwe.Ciphertext) *rlwe.Ciphertext
	if c.requested {
		refresh = func(ct *rlwe.Ciphertext) *rlwe.Ciphertext {
			return c.requestRefresh(crypto.CipherVector{ct})[0]
		}
	}
	eq := equalWithRefresh(c.cp, c.net, c.source, level, in, refresh)
	c.cost.RefreshCiphertexts += eq.Refreshes
	c.cost.RefreshBatches += eq.Refreshes
	if c.owner() {
		c.cost.PlainProducts += 32
		c.cost.CipherProducts += 8
		c.cost.Conjugations++
	}
	return eq
}

func (c *tableCircuit) refresh(v crypto.CipherVector, count int) crypto.CipherVector {
	c.cost.RefreshBatches++
	c.cost.RefreshCiphertexts += count
	if c.requested {
		return c.requestRefresh(v)
	}
	var input crypto.CipherMatrix
	if c.owner() {
		input = crypto.CipherMatrix{v}
	}
	cm := c.net.RefreshCMatWithRandomZero(c.cp, input, c.source, 1, count)
	return cm[0]
}
func (c *tableCircuit) mul(a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	c.cost.CipherProducts++
	return crypto.CMult(c.cp, crypto.CipherVector{a}, crypto.CipherVector{b})[0]
}
func (c *tableCircuit) plain(a *rlwe.Ciphertext, values []complex128) *rlwe.Ciphertext {
	c.cost.PlainProducts++
	return crypto.MulComplexSlots(c.cp, a, values)
}

func (c *tableCircuit) blockSum(a *rlwe.Ciphertext, r, genes int) *rlwe.Ciphertext {
	c.cost.PlainProducts++
	c.cost.Rotations += 2 * (bits.Len(uint(r)) - 1)
	return crypto.BlockSumBroadcast(c.cp, a, r, genes)
}

func (c *tableCircuit) cleanup(x *rlwe.Ciphertext) *rlwe.Ciphertext {
	square := c.mul(x, x)
	linear := crypto.AddConst(c.cp, crypto.CMultConst(c.cp, crypto.CipherVector{x}, -2, false)[0], 3)
	return c.mul(square, linear)
}
func (c *tableCircuit) fourier(a *rlwe.Ciphertext, blockSize, blocks int, inverse bool) *rlwe.Ciphertext {
	c.cost.PlainProducts += 2*blockSize - 1
	c.cost.Rotations += len(crypto.BSGSRotations(1-blockSize, blockSize-1, c.cp.GetSlots()))
	return crypto.BlockFourier(c.cp, a, blockSize, blocks, inverse)
}
func (c *tableCircuit) record(name string, values crypto.CipherVector) {
	if !c.owner() || len(values) == 0 {
		return
	}
	s := TableStage{Name: name, Level: values[0].Level(), MinLogScale: math.Inf(1), MaxLogScale: math.Inf(-1)}
	for _, ct := range values {
		s.Level = min(s.Level, ct.Level())
		s.CiphertextBytes += ct.BinarySize()
		s.MinLogScale = math.Min(s.MinLogScale, math.Log2(ct.Scale.Float64()))
		s.MaxLogScale = math.Max(s.MaxLogScale, math.Log2(ct.Scale.Float64()))
	}
	c.cost.Stages = append(c.cost.Stages, s)
}

type DenseLookupResult struct {
	// [local row][chunk], with each scalar repeated in its gene's M slots.
	Match, Old crypto.CipherMatrix
	Cost       TableCost
}

// LookupDense uses the same immutable incoming table for all M local rows.
// All data sites participate; only source supplies table/keys and receives
// results. Inputs are at maximum level. No destinations or new bits are formed
// here: their site-turn orchestration belongs to E1.
func LookupDense(cp *crypto.CryptoParams, net *mpc.Network, source int, shape DenseShape, table DenseTable, keys [][]string) DenseLookupResult {
	chunks := shape.chunks(cp)
	if net.GetPid() == 0 {
		return DenseLookupResult{}
	}
	c := tableCircuit{cp: cp, net: net, source: source}
	out := DenseLookupResult{}
	if c.owner() {
		if len(keys) != shape.Genes {
			panic("LookupDense: local gene count mismatch")
		}
		out.Match, out.Old = newCipherMatrix(shape.M, chunks), newCipherMatrix(shape.M, chunks)
	}
	for j := 0; j < shape.M; j++ {
		for chunk := 0; chunk < chunks; chunk++ {
			var in EqualityInput
			genes := shape.genes(cp, chunk)
			if c.owner() {
				for d := range in.Query {
					in.Query[d] = make([]complex128, cp.GetSlots())
					in.F[d] = table.F[d][chunk]
				}
				for g := 0; g < genes; g++ {
					local := keys[chunk*(cp.GetSlots()/shape.M)+g]
					if len(local) > shape.M {
						panic("LookupDense: local rows exceed M")
					}
					if j < len(local) {
						code := EncodeKey(local[j])
						for s := g * shape.M; s < (g+1)*shape.M; s++ {
							for d, x := range code {
								in.Query[d][s] = x
							}
						}
					}
				}
			}
			eq := c.equal(cp.Params.MaxLevel(), in)
			// The old path needs a ramp product AND the block-head mask level.
			e := c.refresh(crypto.CipherVector{eq.Match}, 1)[0]
			if c.owner() {
				ramp := make([]complex128, cp.GetSlots())
				for s := 0; s < genes*shape.M; s++ {
					ramp[s] = complex(float64(s%shape.M), 0)
				}
				out.Match[j][chunk] = c.blockSum(e, shape.M, genes)
				out.Old[j][chunk] = c.blockSum(c.plain(e, ramp), shape.M, genes)
				c.record(fmt.Sprintf("lookup row%d chunk%d", j, chunk), crypto.CipherVector{out.Match[j][chunk], out.Old[j][chunk]})
			}
		}
	}
	out.Cost = c.cost
	return out
}

// UpdateDense appends supplied encrypted new bits using the paper's Fourier
// prefix and updates count by sum(new). New[row][chunk] repeats each gene's bit
// across its block and has already received the prescribed new-bit cleanup.
// The supplied fixture/turn guarantees capacity and that dense p equals the
// append position. Dense has no stored D, so p is not an input to this update.
//
// All table/new inputs are at maximum level. No secret-dependent skipping,
// rounding, or extra F cleanup is used. F/O are refreshed for the next lookup;
// Count keeps its carried arithmetic value. Only source receives the table.
func UpdateDense(cp *crypto.CryptoParams, net *mpc.Network, source int, shape DenseShape, table DenseTable, keys [][]string, newRows crypto.CipherMatrix) (DenseTable, TableCost) {
	chunks := shape.chunks(cp)
	if net.GetPid() == 0 {
		return DenseTable{}, TableCost{}
	}
	c := tableCircuit{cp: cp, net: net, source: source}
	var out DenseTable
	if c.owner() {
		if len(keys) != shape.Genes || len(newRows) != shape.M {
			panic("UpdateDense: input shape mismatch")
		}
		out = DenseTableFromMatrix(newCipherMatrix(densePlanes, chunks))
	}
	for chunk := 0; chunk < chunks; chunk++ {
		genes := shape.genes(cp, chunk)
		var prefix crypto.CipherVector
		if c.owner() {
			omegaMinusOne := make([]complex128, cp.GetSlots())
			for s := 0; s < genes*shape.M; s++ {
				omegaMinusOne[s] = cmplx.Rect(1, 2*math.Pi*float64(s%shape.M)/float64(shape.M)) - 1
			}
			prefix = make(crypto.CipherVector, shape.M)
			frontier := c.fourier(table.O[chunk], shape.M, genes, false)
			prefix[0] = crypto.AddConst(cp, c.plain(frontier, omegaMinusOne), 1)
			for j := 1; j < shape.M; j++ {
				prefix[j] = crypto.AddConst(cp, c.plain(newRows[j-1][chunk], omegaMinusOne), 1)
			}
			prefix = crypto.DropLevel(cp, crypto.CipherMatrix{prefix}, cp.Params.MaxLevel()-2)[0]
			c.record("frontier/factors", prefix)
		}
		level := cp.Params.MaxLevel() - 2
		for stride := 1; stride < shape.M; stride *= 2 {
			if !net.CanCollectiveBootstrap(cp, level-1) {
				prefix = c.refresh(prefix, shape.M)
				level = cp.Params.MaxLevel()
			}
			if c.owner() {
				next := append(crypto.CipherVector(nil), prefix...)
				for j := stride; j < shape.M; j++ {
					next[j] = c.mul(prefix[j], prefix[j-stride])
				}
				prefix = crypto.DropLevel(cp, crypto.CipherMatrix{next}, level-1)[0]
				c.record(fmt.Sprintf("prefix stride%d", stride), prefix)
			}
			level--
		}
		// Reserve levels for b*prefix, IDFT, and the plaintext key write.
		if !net.CanCollectiveBootstrap(cp, level-3) {
			prefix = c.refresh(prefix, shape.M)
		}
		var writes crypto.CipherVector
		var occupancy *rlwe.Ciphertext
		if c.owner() {
			writes = make(crypto.CipherVector, shape.M)
			occupancy = table.O[chunk].CopyNew()
			out.Count[chunk] = table.Count[chunk].CopyNew()
			for j := 0; j < shape.M; j++ {
				writes[j] = c.fourier(c.mul(newRows[j][chunk], prefix[j]), shape.M, genes, true)
				occupancy = crypto.Add(cp, occupancy, writes[j])
				out.Count[chunk] = crypto.Add(cp, out.Count[chunk], newRows[j][chunk])
			}
			c.record("IDFT write weights", writes)
			first := chunk * (cp.GetSlots() / shape.M)
			localKeys := keys[first : first+genes]
			for _, local := range localKeys {
				if len(local) > shape.M {
					panic("UpdateDense: local rows exceed M")
				}
			}
			encoded := encodeLocalKeys(localKeys)
			codes := make([]complex128, cp.GetSlots())
			for d := 0; d < 32; d++ {
				out.F[d][chunk] = table.F[d][chunk].CopyNew()
				for j := 0; j < shape.M; j++ {
					clear(codes) // MulComplexSlots encodes synchronously; it retains no slice.
					for g, local := range encoded {
						if j < len(local) {
							x := local[j][d]
							for s := g * shape.M; s < (g+1)*shape.M; s++ {
								codes[s] = x
							}
						}
					}
					out.F[d][chunk] = crypto.Add(cp, out.F[d][chunk], c.plain(writes[j], codes))
				}
			}
		}
		occupancy = c.refresh(crypto.CipherVector{occupancy}, 1)[0]
		var live crypto.CipherVector
		if c.owner() {
			out.O[chunk] = c.cleanup(occupancy)
			live = make(crypto.CipherVector, 33)
			for d := 0; d < 32; d++ {
				live[d] = out.F[d][chunk]
			}
			live[32] = out.O[chunk]
			c.record("updated F/O before refresh", live)
		}
		live = c.refresh(live, 33)
		if c.owner() {
			for d := 0; d < 32; d++ {
				out.F[d][chunk] = live[d]
			}
			out.O[chunk] = live[32]
			c.record("updated F/O ready for lookup", live)
		}
	}
	return out, c.cost
}
