package embed

import (
	"fmt"

	"github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// proposalRanks counts all real rows, including rows whose encrypted active
// bit is zero, and returns the maximum count over stages and cells.
// Repeated addresses in different stages remain separate proposals.
func proposalRanks(addresses [][]int, p Setting) ([][]int, int) {
	ranks := make([][]int, p.H)
	maximum := 0
	for h := 0; h < p.H; h++ {
		ranks[h] = make([]int, len(addresses))
		counts := make([]int, p.Rho)
		for j, row := range addresses {
			cell := row[h]
			ranks[h][j] = counts[cell]
			counts[cell]++
			maximum = max(maximum, counts[cell])
		}
	}
	return ranks, maximum
}

func (c *tableCircuit) oneMinus(a *rlwe.Ciphertext) *rlwe.Ciphertext {
	return crypto.AddConst(c.cp, crypto.Neg(c.cp, a), 1)
}

// inclusiveProducts keeps every public rank/row, including padded identities.
// The returned products are at maximum level, ready for subsequent payload work.
func (c *tableCircuit) inclusiveProducts(v crypto.CipherVector, n, level int) crypto.CipherVector {
	for stride := 1; stride < n; stride *= 2 {
		if !c.net.CanCollectiveBootstrap(c.cp, level-1) {
			v = c.refresh(v, n)
			level = c.cp.Params.MaxLevel()
		}
		if c.owner() {
			next := append(crypto.CipherVector(nil), v...)
			for j := stride; j < n; j++ {
				next[j] = c.mul(v[j], v[j-stride])
			}
			v = crypto.DropLevel(c.cp, crypto.CipherMatrix{next}, level-1)[0]
		}
		level--
	}
	if level < c.cp.Params.MaxLevel() {
		v = c.refresh(v, n)
	}
	return v
}

// UpdateHash implements one intermediate site's update. All data sites call it;
// only source supplies inputs and receives the updated table. Incoming table,
// cleaned newRows and proposed common coordinates must be at maximum level.
// Row matrices are [local row][chunk], with scalars repeated across each R block.
func UpdateHash(cp *crypto.CryptoParams, net *mpc.Network, source int, shape HashShape, table HashTable, keys [][]string, hashKey []byte, newRows, proposedRows crypto.CipherMatrix) (HashTable, TableCost) {
	var addresses [][][]int
	if net.GetPid() == source {
		addresses = make([][][]int, len(keys))
		for g, local := range keys {
			for _, key := range local {
				addresses[g] = append(addresses[g], HashAddresses(hashKey, key, shape.Setting))
			}
		}
	}
	return updateHash(cp, net, source, shape, table, keys, addresses, newRows, proposedRows)
}

func updateHash(cp *crypto.CryptoParams, net *mpc.Network, source int, shape HashShape, table HashTable, keys [][]string, addresses [][][]int, newRows, proposedRows crypto.CipherMatrix) (HashTable, TableCost) {
	chunks := shape.chunks(cp)
	if net.GetPid() == 0 {
		return HashTable{}, TableCost{}
	}
	c := tableCircuit{cp: cp, net: net, source: source, requested: true}
	if !c.owner() {
		c.serveRefreshRequests()
		return HashTable{}, TableCost{}
	}
	defer c.sendRefreshCommand(refreshDone)
	var out crypto.CipherMatrix
	if c.owner() {
		if len(keys) != shape.Genes || len(addresses) != shape.Genes || len(newRows) != shape.M || len(proposedRows) != shape.M {
			panic("UpdateHash: input shape mismatch")
		}
		out = newCipherMatrix(hashPlanes, chunks)
	}
	r := shape.Rho + shape.Sigma
	for chunk := 0; chunk < chunks; chunk++ {
		genes := shape.genes(cp, chunk)
		var state, active, proposed crypto.CipherVector
		var localKeys [][]string
		var localAddr, ranks [][][]int
		proposalCount := 0
		if c.owner() {
			first := chunk * (cp.GetSlots() / r)
			localKeys, localAddr = keys[first:first+genes], addresses[first:first+genes]
			ranks = make([][][]int, genes)
			for g := 0; g < genes; g++ {
				if len(localKeys[g]) > shape.M || len(localAddr[g]) != len(localKeys[g]) {
					panic("UpdateHash: local rows mismatch")
				}
				var count int
				ranks[g], count = proposalRanks(localAddr[g], shape.Setting)
				proposalCount = max(proposalCount, count)
			}
			state = make(crypto.CipherVector, hashPlanes)
			for d, row := range table.Matrix() {
				state[d] = row[chunk]
			}
			active, proposed = make(crypto.CipherVector, shape.M), make(crypto.CipherVector, shape.M)
			for j := 0; j < shape.M; j++ {
				active[j] = newRows[j][chunk]
				proposed[j] = proposedRows[j][chunk]
				state[planeCount] = crypto.Add(cp, state[planeCount], active[j])
			}
		}
		for h := 0; h < shape.H; h++ {
			c.proposeHashStage(shape, genes, h, proposalCount, state, active, proposed, localKeys, localAddr, ranks)
		}
		active = c.refresh(active, shape.M)
		c.appendHashStash(shape, genes, state, active, proposed, localKeys)
		var occupancy *rlwe.Ciphertext
		if c.owner() {
			occupancy = state[planeO]
		}
		occupancy = c.refresh(crypto.CipherVector{occupancy}, 1)[0]
		if c.owner() {
			state[planeO] = c.cleanup(occupancy)
			c.record("hash update before refresh", state)
		}
		state = c.refresh(state, hashPlanes)
		if c.owner() {
			for d := range out {
				out[d][chunk] = state[d]
			}
			c.record("hash update ready for lookup", state)
		}
	}
	if c.owner() {
		return HashTableFromMatrix(out), c.cost
	}
	return HashTable{}, c.cost
}

// proposeHashStage applies one simultaneous stage, retaining each winner until
// both its key and common-coordinate writes are complete. Only source computes.
func (c *tableCircuit) proposeHashStage(shape HashShape, genes, h, proposalCount int, state, active, proposed crypto.CipherVector, keys [][]string, addresses, ranks [][][]int) {
	if proposalCount == 0 {
		return
	}
	cp := c.cp
	r, maxLevel := shape.Rho+shape.Sigma, cp.Params.MaxLevel()
	var a, payload crypto.CipherVector
	// replication makes this forward routing a mask, without rotations.
	mask := func(rank, row int) []complex128 {
		v := make([]complex128, cp.GetSlots())
		for g := 0; g < genes; g++ {
			if row < len(keys[g]) && ranks[g][h][row] == rank {
				v[g*r+addresses[g][row][h]] = 1
			}
		}
		return v
	}
	if c.owner() {
		a, payload = make(crypto.CipherVector, proposalCount), make(crypto.CipherVector, proposalCount)
		for rank := range a {
			for j := 0; j < shape.M; j++ {
				v := mask(rank, j)
				x, p := c.plain(active[j], v), c.plain(proposed[j], v)
				a[rank], payload[rank] = addCiphertext(cp, a[rank], x), addCiphertext(cp, payload[rank], p)
			}
		}
	}
	a = c.refresh(a, proposalCount)
	var prefix crypto.CipherVector
	if c.owner() {
		prefix = make(crypto.CipherVector, len(a))
		for rank := range a {
			prefix[rank] = c.oneMinus(a[rank])
		}
	}
	prefix = c.inclusiveProducts(prefix, proposalCount, maxLevel)
	var winners crypto.CipherVector
	if c.owner() {
		winners = make(crypto.CipherVector, proposalCount)
		free := c.oneMinus(state[planeO])
		for rank := range winners {
			eligible := a[rank]
			if rank > 0 {
				eligible = c.mul(eligible, prefix[rank-1])
			}
			winners[rank] = c.mul(free, eligible)
		}
	}
	winners = c.refresh(winners, proposalCount)
	if c.owner() {
		encoded := encodeLocalKeys(keys)
		codes := make([]complex128, cp.GetSlots())
		for rank, w := range winners {
			state[planeO] = crypto.Add(cp, state[planeO], w)
			state[planeD] = crypto.Add(cp, state[planeD], c.mul(w, payload[rank]))
			for d := 0; d < planeO; d++ {
				clear(codes)
				for g, local := range encoded {
					for j, code := range local {
						if ranks[g][h][j] == rank {
							codes[g*r+addresses[g][j][h]] = code[d]
						}
					}
				}
				state[d] = crypto.Add(cp, state[d], c.plain(w, codes))
			}
		}
		for j := 0; j < shape.M; j++ {
			var won *rlwe.Ciphertext
			for rank, w := range winners {
				won = addCiphertext(cp, won, c.plain(w, mask(rank, j)))
			}
			active[j] = crypto.Add(cp, active[j], crypto.Neg(cp, c.blockSum(won, r, genes)))
		}
		c.record(fmt.Sprintf("proposal stage%d F/O/D", h+1), state[:planeCount])
	}
}
