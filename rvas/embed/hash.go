package embed

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"math/bits"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Hash wire planes; dense omits D and has a different wire layout.
const (
	planeO = 32 + iota
	planeD
	planeCount
	hashPlanes
)

// HashAddresses returns all H proposals, including repeated addresses.
func HashAddresses(hashKey []byte, key string, p Setting) []int {
	if len(hashKey) != 32 || p.Rho <= 0 || p.H <= 0 {
		panic("hash: expected a 32-byte shared key and positive rho,H")
	}
	mac := hmac.New(sha256.New, hashKey)
	addresses := make([]int, p.H)
	var stage [8]byte
	for h := range addresses {
		mac.Reset()
		mac.Write([]byte("secure-rvas/embed/address/v1\x00"))
		binary.BigEndian.PutUint64(stage[:], uint64(h))
		mac.Write(stage[:])
		mac.Write([]byte(key))
		addresses[h] = int(binary.BigEndian.Uint64(mac.Sum(nil)[:8]) % uint64(p.Rho))
	}
	return addresses
}

// HashShape describes the table packing. Lookup and update derive their
// execution sizes from the source site's actual addresses for each chunk.
type HashShape struct {
	Setting
	Genes int
}

func (s HashShape) chunks(cp *crypto.CryptoParams) int {
	r := s.Rho + s.Sigma
	if s.validate() != nil || s.Rho == 0 || r > cp.GetSlots() || s.Genes < 0 {
		panic("hash: invalid public shape")
	}
	perChunk := cp.GetSlots() / r
	return (s.Genes + perChunk - 1) / perChunk
}

func (s HashShape) genes(cp *crypto.CryptoParams, chunk int) int {
	perChunk := cp.GetSlots() / (s.Rho + s.Sigma)
	return min(perChunk, s.Genes-chunk*perChunk)
}

func (s HashShape) stashTile() int {
	if s.Sigma == 0 {
		return 1
	}
	return 1 << bits.Len(uint(s.Sigma-1))
}
func (s HashShape) stashCopies() int { return (s.Rho + s.Sigma) / s.stashTile() }
func (s HashShape) stashGroups() int {
	if s.Sigma == 0 {
		return 0
	}
	return (s.M + s.stashCopies() - 1) / s.stashCopies()
}
func (s HashShape) stashTransform() int {
	return 1 << bits.Len(uint(s.Sigma))
}

// HashRotations uses public power-of-two rotations for copies, extraction and
// broadcast, plus the public giant-step keys for main routing and stash Fourier.
func HashRotations(params ckks.Parameters, p Setting) []uint64 {
	if p.validate() != nil || p.Rho == 0 || p.Rho+p.Sigma > params.MaxSlots() {
		panic("hash: invalid rotation layout")
	}
	out := []uint64{params.GaloisElementForComplexConjugation()}
	seen := map[uint64]bool{out[0]: true}
	for step := 1; step < p.Rho+p.Sigma; step *= 2 {
		for _, sign := range []int{-1, 1} {
			g := params.GaloisElementForRotation(sign * step)
			if !seen[g] {
				out, seen[g] = append(out, g), true
			}
		}
	}
	out = appendRotationKeys(params, out, crypto.BSGSRotations(1-p.Rho, p.M-1, params.MaxSlots()))
	k := (HashShape{Setting: p}).stashTransform()
	if p.Sigma > 0 {
		out = appendRotationKeys(params, out, crypto.BSGSRotations(1-k, k-1, params.MaxSlots()))
	}
	return out
}

type HashTable struct {
	F           [32]crypto.CipherVector
	O, D, Count crypto.CipherVector
}

// Matrix orders the existing ciphertexts as F0..F31,O,D,Count.
func (t HashTable) Matrix() crypto.CipherMatrix {
	cm := make(crypto.CipherMatrix, hashPlanes)
	copy(cm, t.F[:])
	cm[planeO], cm[planeD], cm[planeCount] = t.O, t.D, t.Count
	return cm
}

func HashTableFromMatrix(cm crypto.CipherMatrix) HashTable {
	if len(cm) != hashPlanes {
		panic("hash: expected 32 key planes, O,D,count")
	}
	var t HashTable
	copy(t.F[:], cm[:planeO])
	t.O, t.D, t.Count = cm[planeO], cm[planeD], cm[planeCount]
	return t
}

func localHashAddresses(keys [][]string, p Setting, hashKey []byte) [][][]int {
	out := make([][][]int, len(keys))
	for g, rows := range keys {
		if len(rows) > p.M {
			panic("hash: local rows exceed M")
		}
		out[g] = make([][]int, len(rows))
		for j, key := range rows {
			out[g][j] = HashAddresses(hashKey, key, p)
		}
	}
	return out
}

// hashAnchorRows applies stages in order, with local-row priority inside each
// stage.
func hashAnchorRows(p Setting, addresses [][]int) []int {
	rows := make([]int, p.Rho+p.Sigma)
	for s := range rows {
		rows[s] = -1
	}
	active := make([]bool, len(addresses))
	for j := range active {
		active[j] = true
	}
	for h := 0; h < p.H; h++ {
		for j, hs := range addresses {
			cell := hs[h]
			if active[j] && rows[cell] == -1 {
				rows[cell], active[j] = j, false
			}
		}
	}
	next := p.Rho
	for j, remaining := range active {
		if remaining {
			if next == len(rows) {
				return rows
			}
			rows[next], next = j, next+1
		}
	}
	return rows
}

// BuildHashAnchor is local to P1, under the collective public key. Count
// reserves every anchor coordinate.
func BuildHashAnchor(cp *crypto.CryptoParams, shape HashShape, keys [][]string, hashKey []byte) HashTable {
	return buildHashAnchor(cp, shape, keys, localHashAddresses(keys, shape.Setting, hashKey))
}

func buildHashAnchor(cp *crypto.CryptoParams, shape HashShape, keys [][]string, addresses [][][]int) HashTable {
	chunks, r := shape.chunks(cp), shape.Rho+shape.Sigma
	if len(keys) != shape.Genes || len(addresses) != shape.Genes {
		panic("hash: local gene count mismatch")
	}
	cm := newCipherMatrix(hashPlanes, chunks)
	for chunk := 0; chunk < chunks; chunk++ {
		values := newPlanes(hashPlanes, cp.GetSlots())
		for g := 0; g < shape.genes(cp, chunk); g++ {
			idx := chunk*(cp.GetSlots()/r) + g
			rows := hashAnchorRows(shape.Setting, addresses[idx])
			for s, j := range rows {
				if j >= 0 {
					for d, x := range EncodeKey(keys[idx][j]) {
						values[d][g*r+s] = x
					}
					values[planeO][g*r+s] = 1
					values[planeD][g*r+s] = complex(float64(j), 0)
				}
				values[planeCount][g*r+s] = complex(float64(len(keys[idx])), 0)
			}
		}
		encryptPlanes(cp, cm, chunk, values)
	}
	return HashTableFromMatrix(cm)
}

type hashPair struct{ row, cell int }

// hashGroups colors the deduplicated bipartite query/cell graph into exactly
// its maximum degree many matchings, using two-color alternating paths.
// Pairs stay at the querying site; the maximum group count across a chunk
// determines its main lookup schedule.
func hashGroups(addresses [][]int, rho int) [][]hashPair {
	left := len(addresses)
	degree := make([]int, left+rho)
	var edges []hashPair
	for j, hs := range addresses {
		seen := make(map[int]bool)
		for _, cell := range hs {
			if !seen[cell] {
				edges = append(edges, hashPair{j, cell})
				degree[j]++
				degree[left+cell]++
				seen[cell] = true
			}
		}
	}
	delta := 0
	for _, d := range degree {
		delta = max(delta, d)
	}
	links := make([][]int, len(degree))
	for v := range links {
		links[v] = make([]int, delta)
		for color := range links[v] {
			links[v][color] = -1
		}
	}
	missing := func(v int) int {
		for color := 0; color < delta; color++ {
			if links[v][color] == -1 {
				return color
			}
		}
		panic("hash: no free edge color")
	}
	for _, edge := range edges {
		a, b := edge.row, left+edge.cell
		alpha, beta := missing(a), missing(b)
		if alpha != beta {
			type pathEdge struct{ a, b, color int }
			var path []pathEdge
			v, color := a, beta
			for links[v][color] != -1 {
				other := links[v][color]
				path = append(path, pathEdge{v, other, color})
				v = other
				if color == beta {
					color = alpha
				} else {
					color = beta
				}
			}
			for _, e := range path {
				links[e.a][e.color], links[e.b][e.color] = -1, -1
			}
			for _, e := range path {
				c := alpha
				if e.color == alpha {
					c = beta
				}
				links[e.a][c], links[e.b][c] = e.b, e.a
			}
		}
		links[a][beta], links[b][beta] = b, a
	}
	groups := make([][]hashPair, delta)
	for j := 0; j < left; j++ {
		for color := 0; color < delta; color++ {
			if cell := links[j][color]; cell != -1 {
				groups[color] = append(groups[color], hashPair{j, cell - left})
			}
		}
	}
	return groups
}

// rotate decomposes only public displacements into the public power-of-two
// key set. Each called rotation is included in the arithmetic cost.
func (c *tableCircuit) rotate(x *rlwe.Ciphertext, shift int) *rlwe.Ciphertext {
	sign := 1
	if shift < 0 {
		sign, shift = -1, -shift
	}
	for step := 1; shift > 0; step *= 2 {
		if shift&1 != 0 {
			x = crypto.RotateRight(c.cp, x, sign*step)
			c.cost.Rotations++
		}
		shift >>= 1
	}
	return x
}

func (c *tableCircuit) stashCopies(x *rlwe.Ciphertext, shape HashShape, genes int) *rlwe.Ciphertext {
	r, tile := shape.Rho+shape.Sigma, shape.stashTile()
	mask := make([]complex128, c.cp.GetSlots())
	for g := 0; g < genes; g++ {
		for s := 0; s < shape.Sigma; s++ {
			mask[g*r+s] = 1
		}
	}
	x = c.plain(c.rotate(x, -shape.Rho), mask)
	for step := tile; step < r; step *= 2 {
		x = crypto.Add(c.cp, x, c.rotate(x, step))
	}
	return x
}

// mainRows uses BSGS for every public diagonal, including zero masks.
func (c *tableCircuit) mainRows(x *rlwe.Ciphertext, shape HashShape, pairs [][]hashPair) *rlwe.Ciphertext {
	r := shape.Rho + shape.Sigma
	c.cost.PlainProducts += shape.Rho + shape.M - 1
	c.cost.Rotations += len(crypto.BSGSRotations(1-shape.Rho, shape.M-1, c.cp.GetSlots()))
	return crypto.DiagonalBSGS(c.cp, x, 1-shape.Rho, shape.M-1, func(shift int, mask []complex128) {
		for g, edges := range pairs {
			for _, edge := range edges {
				if edge.row-edge.cell == shift {
					mask[g*r+edge.row] = 1
				}
			}
		}
	})
}

func (c *tableCircuit) stashRows(x *rlwe.Ciphertext, shape HashShape, genes, first int) *rlwe.Ciphertext {
	r, tile := shape.Rho+shape.Sigma, shape.stashTile()
	for step := 1; step < tile; step *= 2 {
		x = crypto.Add(c.cp, x, c.rotate(x, -step))
	}
	var out *rlwe.Ciphertext
	for copy := 0; copy < shape.stashCopies() && first+copy < shape.M; copy++ {
		mask := make([]complex128, c.cp.GetSlots())
		for g := 0; g < genes; g++ {
			mask[g*r+first+copy] = 1
		}
		out = addCiphertext(c.cp, out, c.plain(c.rotate(x, first+copy-copy*tile), mask))
	}
	return out
}

func (c *tableCircuit) conjugate(x *rlwe.Ciphertext) *rlwe.Ciphertext {
	out := conjugate(c.cp, x)
	c.cost.Conjugations++
	return out
}

// packHashMatch carries 2*(Re(e) + i*Re(e)*D) through one real linear map.
// Project e first to avoid amplifying its imaginary error by D. The factor 2
// is removed in the existing output masks, without a separate rescale.
func (c *tableCircuit) packHashMatch(e, payload *rlwe.Ciphertext) *rlwe.Ciphertext {
	realTwice := crypto.Add(c.cp, e, c.conjugate(e))
	imaginaryPayload := crypto.CMultConst(c.cp, crypto.CipherVector{payload}, 1i, false)[0]
	return c.mul(realTwice, crypto.AddConst(c.cp, imaginaryPayload, 1))
}

// unpackHashRows splits a refreshed sum of packed matches. Its two factors of
// 2 (projection before packing and conjugate splitting here) give head weights
// 1/4 and -i/4. The component's existing [row][chunk] output layout is retained.
func (c *tableCircuit) unpackHashRows(packed *rlwe.Ciphertext, shape HashShape, genes int) (match, old crypto.CipherVector) {
	r := shape.Rho + shape.Sigma
	conjugate := c.conjugate(packed)
	parts := crypto.CipherVector{crypto.Add(c.cp, packed, conjugate), crypto.Add(c.cp, packed, crypto.Neg(c.cp, conjugate))}
	match, old = make(crypto.CipherVector, shape.M), make(crypto.CipherVector, shape.M)
	rows := []crypto.CipherVector{match, old}
	weights := []complex128{0.25, -0.25i}
	for which, part := range parts {
		heads := make([]complex128, c.cp.GetSlots())
		for g := 0; g < genes; g++ {
			heads[g*r] = weights[which]
		}
		for j := 0; j < shape.M; j++ {
			x := c.plain(c.rotate(part, -j), heads)
			for step := 1; step < r; step *= 2 {
				x = crypto.Add(c.cp, x, c.rotate(x, step))
			}
			rows[which][j] = x
		}
	}
	return
}

type HashLookupResult struct {
	// Same component contract as dense: [local row][chunk], scalar repeated
	// in each gene's R slots. This does not choose E2's final C2S layout.
	Match, Old crypto.CipherMatrix
	Cost       TableCost
}

// LookupHash reads one immutable incoming table. Only source supplies local
// keys/table/hashKey and receives the result. It computes the minimum main
// group count per gene, then uses the chunk maximum locally. Other data sites
// service refresh requests until the lookup completes.
// Input planes use full slots at maximum level.
func LookupHash(cp *crypto.CryptoParams, net *mpc.Network, source int, shape HashShape, table HashTable, keys [][]string, hashKey []byte) HashLookupResult {
	var addresses [][][]int
	if net.GetPid() == source {
		addresses = localHashAddresses(keys, shape.Setting, hashKey)
	}
	return lookupHash(cp, net, source, shape, table, keys, addresses)
}

func lookupHash(cp *crypto.CryptoParams, net *mpc.Network, source int, shape HashShape, table HashTable, keys [][]string, addresses [][][]int) HashLookupResult {
	chunks, r, tile := shape.chunks(cp), shape.Rho+shape.Sigma, shape.stashTile()
	if net.GetPid() == 0 {
		return HashLookupResult{}
	}
	c := tableCircuit{cp: cp, net: net, source: source, requested: true}
	if !c.owner() {
		c.serveRefreshRequests()
		return HashLookupResult{}
	}
	defer c.sendRefreshCommand(refreshDone)
	out := HashLookupResult{}
	if c.owner() {
		if len(keys) != shape.Genes || len(addresses) != shape.Genes {
			panic("hash: local gene count mismatch")
		}
		out.Match, out.Old = newCipherMatrix(shape.M, chunks), newCipherMatrix(shape.M, chunks)
	}
	for chunk := 0; chunk < chunks; chunk++ {
		genes := shape.genes(cp, chunk)
		lookupGroups := 0
		var groups [][][]hashPair
		var stashF [32]*rlwe.Ciphertext
		var stashD, packed *rlwe.Ciphertext
		if c.owner() {
			groups = make([][][]hashPair, genes)
			for g := 0; g < genes; g++ {
				groups[g] = hashGroups(addresses[chunk*(cp.GetSlots()/r)+g], shape.Rho)
				lookupGroups = max(lookupGroups, len(groups[g]))
			}
			if shape.Sigma > 0 {
				for d := range stashF {
					stashF[d] = c.stashCopies(table.F[d][chunk], shape, genes)
				}
				stashD = c.stashCopies(table.D[chunk], shape, genes)
			}
		}
		for group := 0; group < lookupGroups+shape.stashGroups(); group++ {
			main := group < lookupGroups
			first := (group - lookupGroups) * shape.stashCopies()
			in := EqualityInput{}
			var pairs [][]hashPair
			if c.owner() {
				pairs = make([][]hashPair, genes)
				for d := range in.Query {
					in.Query[d] = make([]complex128, cp.GetSlots())
					in.F[d] = stashF[d]
					if main {
						in.F[d] = table.F[d][chunk]
					}
				}
				for g := 0; g < genes; g++ {
					local := keys[chunk*(cp.GetSlots()/r)+g]
					if main {
						if group < len(groups[g]) {
							pairs[g] = groups[g][group]
						}
					} else {
						for copy := 0; copy < shape.stashCopies() && first+copy < len(local); copy++ {
							for s := 0; s < shape.Sigma; s++ {
								pairs[g] = append(pairs[g], hashPair{first + copy, copy*tile + s})
							}
						}
					}
					for _, pair := range pairs[g] {
						for d, x := range EncodeKey(local[pair.row]) {
							in.Query[d][g*r+pair.cell] = x
						}
					}
				}
			}
			level := cp.Params.MaxLevel()
			if !main {
				level-- // Stash extraction mask; copies/rotations use no levels.
			}
			eq := c.equal(level, in)
			e := c.refresh(crypto.CipherVector{eq.Match}, 1)[0]
			if c.owner() {
				var routed *rlwe.Ciphertext
				if main {
					routed = c.mainRows(c.packHashMatch(e, table.D[chunk]), shape, pairs)
				} else {
					routed = c.stashRows(c.packHashMatch(e, stashD), shape, genes, first)
				}
				packed = addCiphertext(cp, packed, routed)
			}
		}
		if c.owner() && packed == nil {
			// An empty chunk with no stash still returns encrypted zero rows.
			zero, _ := crypto.EncryptComplexVector(cp, make([]complex128, cp.GetSlots()))
			packed = zero[0]
		}
		// Reserve levels before exposing per-row values to later components.
		compact := c.refresh(crypto.CipherVector{packed}, 1)[0]
		if c.owner() {
			match, old := c.unpackHashRows(compact, shape, genes)
			for j := 0; j < shape.M; j++ {
				out.Match[j][chunk], out.Old[j][chunk] = match[j], old[j]
			}
			c.record("hash match/old", crypto.CipherVector{out.Match[0][chunk], out.Old[0][chunk]})
		}
	}
	if c.owner() {
		out.Cost = c.cost
	}
	return out
}
