package embed

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/hhcho/sfgwas/rvas/cap"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// PipelineConfig is public and identical at all parties. Mode selects a
// dense/hash or cap-dependent auto policy; Settings permits explicit experiments.
type PipelineConfig struct {
	Genes    []string
	Cap      cap.Params
	CKKS     string
	Settings map[int]Setting
	Mode     string
	DenseMax int // auto uses dense for caps up to and including this value
	Hash     HashPolicy
	Caps     []int // optional previously revealed caps, in Genes order; skips private cap
}

type EmbeddingBatch struct {
	Setting Setting
	Genes   []int // indices in the original public gene order
}

// PipelineResult contains public planning metadata and this party's E shares.
// E[site-1][gene][row][column] follows Config.Genes, even across different caps.
// P0's E is zero. No coordinates or reconstructed E are released.
type PipelineResult struct {
	Caps           []int
	Batches        []EmbeddingBatch
	Galois         []uint64
	GaloisKeyBytes int // serialized evaluation keys held by this party
	E              [][]mpc_core.RMat
}

// RunEmbedding runs the variant embedding after MPC session initialization, with P1 as hub.
// Each site supplies only its own unique keys per gene; P0 supplies nil.
// The selected cap must hold the true union and J must cover the estimate, as
// required by the cap contract.
func RunEmbedding(m *mpc.MPC, config PipelineConfig, keys [][]string) (PipelineResult, error) {
	return RunEmbeddingObserved(m, config, keys, PipelineObserver{})
}

// PipelineObserver measures local elapsed time without protocol barriers.
// Callbacks must not communicate or mutate public caps. CoordinateStep returns
// a function that ends a step; it includes any collective waits in that step.
type PipelineObserver struct {
	Phase          func(string)
	Caps           func([]int)
	CoordinateStep func(m int, stage string) func()
}

// RunEmbeddingObserved adds local phase boundaries for measurement. Callbacks
// must not communicate or inspect private values. An empty Phase closes timing.
// Boundaries are not barriers: P0 skips E1 and may wait in E2 for the data sites.
// Caps receives the public caps once, including when reused, before HE setup.
func RunEmbeddingObserved(m *mpc.MPC, config PipelineConfig, keys [][]string, observer PipelineObserver) (PipelineResult, error) {
	phase := func(name string) {
		if observer.Phase != nil {
			observer.Phase(name)
		}
	}
	phase("prepare")
	defer phase("")
	var out PipelineResult
	var params ckks.Parameters
	literal, err := crypto.ResolveCKKSParametersLiteral(config.CKKS)
	if err == nil {
		params, err = ckks.NewParametersFromLiteral(literal)
	}
	if err == nil {
		err = config.validate(params.MaxSlots())
	}
	if m.GetNParty() < 3 || m.GetHubPid() != 1 {
		return out, fmt.Errorf("embedding requires P0 and at least two sites, with hub P1")
	}
	// A fixed-size agreement precedes variable-size cap communication. Include
	// MPC precision and the public gene order, but never hash private key lists.
	public, marshalErr := json.Marshal(struct {
		Config                      PipelineConfig
		Parties, DataBits, FracBits int
	}{config, m.GetNParty(), m.GetDataBits(), m.GetFracBits()})
	localOK := err == nil && marshalErr == nil
	if m.GetPid() > 0 {
		localOK = localOK && len(keys) == len(config.Genes)
		for g, rows := range keys {
			if config.Caps != nil && g < len(config.Caps) {
				localOK = localOK && len(rows) <= config.Caps[g]
			}
			seen := make(map[string]bool, len(rows))
			for _, key := range rows {
				localOK = localOK && !seen[key]
				seen[key] = true
			}
		}
	}
	if !agreePipeline(m, sha256.Sum256(public), localOK) {
		if err != nil {
			return out, err
		}
		return out, fmt.Errorf("embedding: public configuration mismatch or invalid local input shape/duplicate keys")
	}
	out.Caps = append([]int(nil), config.Caps...)
	if config.Caps == nil {
		phase("private_cap")
		capKey := m.DeriveHashKey()
		thresholds := cap.BuildThresholds(config.Cap)
		out.Caps = make([]int, len(config.Genes))
		// Keep the existing workflow's cap batching bound; this is independent of
		// HE packing, which is determined only after the caps have been revealed.
		for first := 0; first < len(config.Genes); first += 64 {
			end := min(first+64, len(config.Genes))
			var bitmaps [][]uint8
			if m.GetPid() > 0 {
				for _, rows := range keys[first:end] {
					bitmaps = append(bitmaps, cap.LocalBitmap(config.Cap, capKey, rows))
				}
			}
			indices := cap.SecureCaps(m, config.Cap, thresholds, end-first, bitmaps)
			for j, index := range indices {
				out.Caps[first+j] = thresholds.Ladder[index]
			}
		}
	}
	phase("table_key")
	if observer.Caps != nil {
		observer.Caps(out.Caps)
	}
	tableKey := m.DeriveHashKey()
	phase("plan")
	groups := make(map[int][]int)
	for g, size := range out.Caps {
		groups[size] = append(groups[size], g)
	}
	sizes := make([]int, 0, len(groups))
	for size := range groups {
		sizes = append(sizes, size)
	}
	sort.Ints(sizes)
	rotations := make(map[uint64]bool)
	for _, size := range sizes {
		p, err := config.setting(size, params.MaxSlots())
		if err != nil {
			return out, err
		}
		out.Batches = append(out.Batches, EmbeddingBatch{p, groups[size]})
		for _, g := range CoordinateRotations(params, p) {
			rotations[g] = true
		}
	}
	for g := range rotations {
		out.Galois = append(out.Galois, g)
	}
	sort.Slice(out.Galois, func(i, j int) bool { return out.Galois[i] < out.Galois[j] })
	out.E = make([][]mpc_core.RMat, m.GetNParty()-1)
	for site := range out.E {
		out.E[site] = make([]mpc_core.RMat, len(config.Genes))
	}
	if len(out.Batches) == 0 {
		return out, nil
	}
	phase("keygen")
	cp := (mpc.ParallelNetworks{m.Network}).CollectiveInit(&params, 53, true, out.Galois)
	for _, key := range cp.RotKs {
		out.GaloisKeyBytes += key.BinarySize()
	}
	for _, batch := range out.Batches {
		// Keep intermediate ciphertexts bounded to one table chunk, even when
		// a cap group spans many chromosomes. E shares are retained for statistics.
		perChunk := params.MaxSlots() / (batch.Setting.Rho + batch.Setting.Sigma)
		for first := 0; first < len(batch.Genes); first += perChunk {
			genes := batch.Genes[first:min(first+perChunk, len(batch.Genes))]
			var local [][]string
			if m.GetPid() > 0 {
				for _, g := range genes {
					local = append(local, keys[g])
				}
			}
			phase(fmt.Sprintf("M%d_coordinates", batch.Setting.M))
			var step func(string) func()
			if observer.CoordinateStep != nil {
				step = func(name string) func() { return observer.CoordinateStep(batch.Setting.M, name) }
			}
			coordinates := assignCoordinatesTimed(cp, m.Network, batch.Setting, len(genes), local, tableKey, step)
			phase(fmt.Sprintf("M%d_finalize", batch.Setting.M))
			e := FinalizeEmbedding(cp, m, batch.Setting, len(genes), coordinates)
			for site := range e {
				for j, g := range genes {
					out.E[site][g] = e[site][j]
				}
			}
		}
	}
	return out, nil
}

func (c PipelineConfig) setting(size, slots int) (Setting, error) {
	if c.Mode != "" {
		return TableSetting(c.Mode, size, slots, c.Hash, c.DenseMax)
	}
	if p, ok := c.Settings[size]; ok {
		return p, nil
	}
	return Setting{}, fmt.Errorf("embedding: no explicit table setting for selected cap %d", size)
}

func (c PipelineConfig) validate(slots int) error {
	if c.Mode != "" {
		if len(c.Settings) != 0 {
			return fmt.Errorf("embedding: use Mode or explicit Settings, not both")
		}
		if _, err := TableSetting(c.Mode, 1, slots, c.Hash, c.DenseMax); err != nil {
			return err
		}
	}
	if c.Caps != nil {
		if len(c.Caps) != len(c.Genes) {
			return fmt.Errorf("embedding: cap count differs from gene count")
		}
		for _, size := range c.Caps {
			if size < 1 || size&(size-1) != 0 || size > slots {
				return fmt.Errorf("embedding: invalid reused cap %d", size)
			}
		}
	}
	p := c.Cap
	if p.LogMu < 1 || p.LogMu > 30 || p.J < 1 || p.J > 30 ||
		p.Floor < 1 || p.Floor&(p.Floor-1) != 0 || p.Floor > int(^uint(0)>>1)>>uint(p.J) ||
		p.Epsilon < 0 || math.IsNaN(p.Epsilon) || math.IsInf(p.Epsilon, 0) {
		return fmt.Errorf("embedding: invalid cap parameters")
	}
	seen := make(map[string]bool)
	for _, gene := range c.Genes {
		if seen[gene] {
			return fmt.Errorf("embedding: duplicate public gene %q", gene)
		}
		seen[gene] = true
	}
	for size, setting := range c.Settings {
		if setting.M != size || setting.validate() != nil || setting.Rho+setting.Sigma > slots {
			return fmt.Errorf("embedding: invalid setting for cap %d", size)
		}
	}
	return nil
}

func agreePipeline(m *mpc.MPC, digest [32]byte, valid bool) bool {
	message := make([]uint64, 5)
	for j := 0; j < 4; j++ {
		message[j] = binary.LittleEndian.Uint64(digest[j*8:])
	}
	if valid {
		message[4] = 1
	}
	if m.GetPid() != 1 {
		m.Network.SendIntVector(message, 1)
		return m.Network.ReceiveInt(1) == 1
	}
	ok := message[4]
	for pid := 0; pid < m.GetNParty(); pid++ {
		if pid == 1 {
			continue
		}
		other := m.Network.ReceiveIntVector(5, pid)
		for j := 0; j < 4; j++ {
			if other[j] != message[j] {
				ok = 0
			}
		}
		ok &= other[4]
	}
	for pid := 0; pid < m.GetNParty(); pid++ {
		if pid != 1 {
			m.Network.SendInt(int(ok), pid)
		}
	}
	return ok == 1
}
