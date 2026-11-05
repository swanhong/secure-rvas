package embed

import (
	"fmt"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// CoordinateResult is the intermediate E1 output, before C2S/Recover/Decode.
// Destinations belong only to the calling site, in [row][chunk] order; each
// gene's scalar is repeated across its R slots. P0/P1 have no destinations:
// P1's eventual embedding is the identity, not a decoded coordinate matrix.
type CoordinateResult struct {
	Destinations crypto.CipherMatrix
}

// CoordinateRotations gives the keys needed by the selected table circuit.
func CoordinateRotations(params ckks.Parameters, p Setting) []uint64 {
	if p.validate() != nil || p.Rho+p.Sigma > params.MaxSlots() {
		panic("coordinates: invalid setting")
	}
	if p.Rho > 0 {
		return HashRotations(params, p)
	}
	out := []uint64{params.GaloisElementForComplexConjugation()}
	for step := 1; step < p.M; step *= 2 {
		out = append(out, params.GaloisElementForRotation(step), params.GaloisElementForRotation(-step))
	}
	return appendRotationKeys(params, out, crypto.BSGSRotations(1-p.M, p.M-1, params.MaxSlots()))
}

// steps are right rotations, while Lattigo names keys by left rotations.
func appendRotationKeys(params ckks.Parameters, keys []uint64, steps []int) []uint64 {
	seen := make(map[uint64]bool, len(keys))
	for _, g := range keys {
		seen[g] = true
	}
	for _, step := range steps {
		g := params.GaloisElementForRotation(-step)
		if !seen[g] {
			keys, seen[g] = append(keys, g), true
		}
	}
	return keys
}

// AssignCoordinates runs P1's anchor and the P2..PN turns under an existing
// collective key. All data sites call it with identical p/genes, but supply
// only their OWN keys[gene][real row], in local order (unique within a gene).
// The union must fit M. Hash sites use the same session hashKey. Dense ignores
// hashKey. P0 may call with nil inputs and does not participate in MHE work.
// No values are decrypted, rounded, or decoded into E here.
func AssignCoordinates(cp *crypto.CryptoParams, net *mpc.Network, p Setting, genes int, keys [][]string, hashKey []byte) CoordinateResult {
	return assignCoordinatesTimed(cp, net, p, genes, keys, hashKey, nil)
}

func assignCoordinatesTimed(cp *crypto.CryptoParams, net *mpc.Network, p Setting, genes int, keys [][]string, hashKey []byte, timing func(string) func()) CoordinateResult {
	done := coordinateTimer(timing, "prepare")
	var addresses [][][]int
	if net.GetPid() > 0 && p.Rho > 0 {
		addresses = localHashAddresses(keys, p, hashKey)
	}
	done()
	return assignCoordinates(cp, net, p, genes, keys, addresses, nil, timing)
}

func coordinateTimer(timing func(string) func(), name string) func() {
	if timing != nil {
		return timing(name)
	}
	return func() {}
}

// Internal observation points let synthetic tests inspect the real circuit.
// The observer runs at every data site and must not mutate its ciphertexts.
// Row values exist only at Source; table snapshots exist only at TableOwner.
type coordinateStep struct {
	Source                                  int
	TableOwner                              int
	Stage                                   string // anchor, coordinates, update, final
	Table                                   crypto.CipherMatrix
	Match, Old, New, Proposed, Destinations crypto.CipherMatrix
}

func assignCoordinates(cp *crypto.CryptoParams, net *mpc.Network, p Setting, genes int, keys [][]string, addresses [][][]int, observe func(coordinateStep), timing func(string) func()) CoordinateResult {
	var result CoordinateResult
	if net.GetPid() == 0 {
		return result
	}
	if p.validate() != nil || net.GetNParty() < 3 || len(keys) != genes {
		panic("coordinates: invalid setting/site count/local gene count")
	}
	for _, rows := range keys {
		if len(rows) > p.M {
			panic("coordinates: local rows exceed M")
		}
	}
	// The common row layout uses R slots per gene, including for hash.
	layout := DenseShape{M: p.Rho + p.Sigma, Genes: genes}
	chunks := layout.chunks(cp)
	if chunks == 0 {
		return result
	}
	dense := DenseShape{M: p.M, Genes: genes}
	hash := HashShape{Setting: p, Genes: genes}
	planes := densePlanes
	if p.Rho > 0 {
		planes = hashPlanes
	}
	emit := func(step coordinateStep) {
		if observe != nil {
			observe(step)
		}
	}
	var table crypto.CipherMatrix
	done := coordinateTimer(timing, "anchor")
	if net.GetPid() == 1 {
		if p.Rho == 0 {
			table = BuildDenseAnchor(cp, dense, keys).Matrix()
		} else {
			table = buildHashAnchor(cp, hash, keys, addresses).Matrix()
		}
	}
	done()
	done = coordinateTimer(timing, "anchor_transfer")
	table = net.TransferCMatWithRandomZero(cp, table, 1, 2, planes, chunks)
	done()
	emit(coordinateStep{Source: 1, TableOwner: 2, Stage: "anchor", Table: table})
	last := net.GetNParty() - 1
	for source := 2; source <= last; source++ {
		prefix := fmt.Sprintf("site%d_", source)
		c := tableCircuit{cp: cp, net: net, source: source}
		var match, old crypto.CipherMatrix
		var count crypto.CipherVector
		var denseTable DenseTable
		var hashTable HashTable
		done = coordinateTimer(timing, prefix+"lookup")
		if p.Rho == 0 {
			if c.owner() {
				denseTable = DenseTableFromMatrix(table)
			}
			lookup := LookupDense(cp, net, source, dense, denseTable, keys)
			match, old, count = lookup.Match, lookup.Old, denseTable.Count
		} else {
			if c.owner() {
				hashTable = HashTableFromMatrix(table)
			}
			lookup := lookupHash(cp, net, source, hash, hashTable, keys, addresses)
			match, old, count = lookup.Match, lookup.Old, hashTable.Count
		}
		done()
		done = coordinateTimer(timing, prefix+"assign")
		newBits, proposed, delta := c.assignRows(p.M, layout, keys, match, old, count)
		done()
		if c.owner() {
			result.Destinations = delta
		}
		emit(coordinateStep{Source: source, Stage: "coordinates", Match: match, Old: old, New: newBits, Proposed: proposed, Destinations: delta})
		if source == last {
			break // Last-site new keys receive coordinates but need no table write.
		}
		var updated crypto.CipherMatrix
		done = coordinateTimer(timing, prefix+"update")
		if p.Rho == 0 {
			t, _ := UpdateDense(cp, net, source, dense, denseTable, keys, newBits)
			updated = t.Matrix()
		} else {
			t, _ := updateHash(cp, net, source, hash, hashTable, keys, addresses, newBits, proposed)
			updated = t.Matrix()
		}
		done()
		done = coordinateTimer(timing, prefix+"transfer")
		table = net.TransferCMatWithRandomZero(cp, updated, source, source+1, planes, chunks)
		done()
		emit(coordinateStep{Source: source, TableOwner: source + 1, Stage: "update", Table: table})
	}
	emit(coordinateStep{Source: last, TableOwner: last, Stage: "final", Table: table})
	return result
}

// assignRows shares the paper's coordinate formulas between dense and hash.
// Real-row masks vary by gene. Adding the full slot vector (not a scalar 1)
// excludes dummy rows from novelty in the ideal formulas without mixing gene blocks.
func (c *tableCircuit) assignRows(m int, layout DenseShape, keys [][]string, match, old crypto.CipherMatrix, count crypto.CipherVector) (newBits, proposed, delta crypto.CipherMatrix) {
	cp := c.cp
	chunks := layout.chunks(cp)
	if c.owner() {
		newBits = newCipherMatrix(m, chunks)
		proposed, delta = newCipherMatrix(m, chunks), newCipherMatrix(m, chunks)
	}
	for chunk := 0; chunk < chunks; chunk++ {
		var cleaned crypto.CipherVector
		if c.owner() {
			cleaned = make(crypto.CipherVector, m)
			for j := 0; j < m; j++ {
				mask := make([]complex128, cp.GetSlots())
				for g := 0; g < layout.genes(cp, chunk); g++ {
					local := keys[chunk*(cp.GetSlots()/layout.M)+g]
					if j < len(local) {
						for s := g * layout.M; s < (g+1)*layout.M; s++ {
							mask[s] = 1
						}
					}
				}
				x := crypto.AddConst(cp, crypto.Neg(cp, match[j][chunk]), mask)
				cleaned[j] = c.cleanup(x)
			}
		}
		cleaned = c.refresh(cleaned, m)
		if c.owner() {
			next := count[chunk]
			for j := 0; j < m; j++ {
				newBits[j][chunk] = cleaned[j]
				proposed[j][chunk] = next
				delta[j][chunk] = crypto.Add(cp, old[j][chunk], c.mul(cleaned[j], next))
				next = crypto.Add(cp, next, cleaned[j])
			}
		}
	}
	return
}
