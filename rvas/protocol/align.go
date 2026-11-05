package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

// AlignShares returns Σ_k E_kᵀ x_k per gene at fixed-point scale.
// embeddings[site-1][gene] matches embed.PipelineResult.E; E_1 is identity.
// Each site supplies its local padded matrices, and P0 passes nil.
// E contains unscaled integer bits, so alignment needs no truncation.
func AlignShares(mpcObj *mpc.MPC, embeddings [][]mpc_core.RMat, local []*mat.Dense, columns int) []mpc_core.RMat {
	if len(embeddings) != mpcObj.GetNParty()-1 {
		panic("AlignShares: expected embeddings for every data site")
	}
	shapes := make([][2]int, len(embeddings[0]))
	for g, embedding := range embeddings[0] {
		shapes[g] = [2]int{len(embedding), columns}
	}
	return alignShared(mpcObj, embeddings, shareSites(mpcObj, local, shapes))
}

// shareSites returns fixed-point shares indexed by [site-1][gene], using
// one owner input per site for all genes. P0 passes nil local matrices.
func shareSites(mpcObj *mpc.MPC, local []*mat.Dense, shapes [][2]int) [][]mpc_core.RMat {
	rtype, fracBits := mpcObj.GetRType(), mpcObj.GetFracBits()
	total := 0
	for _, shape := range shapes {
		total += shape[0] * shape[1]
	}
	var flat mpc_core.RVec
	if mpcObj.GetPid() != auxiliaryPartyID {
		flat = make(mpc_core.RVec, 0, total)
		for g, shape := range shapes {
			for row := 0; row < shape[0]; row++ {
				for column := 0; column < shape[1]; column++ {
					flat = append(flat, rtype.FromFloat64(local[g].At(row, column), fracBits))
				}
			}
		}
	}
	sites := make([][]mpc_core.RMat, mpcObj.GetNParty()-1)
	for site := range sites {
		input := mpcObj.ShareFromOwner(site+1, flat, total)
		sites[site] = make([]mpc_core.RMat, len(shapes))
		offset := 0
		for g, shape := range shapes {
			sites[site][g] = make(mpc_core.RMat, shape[0])
			for row := range sites[site][g] {
				end := offset + shape[1]
				sites[site][g][row] = input[offset:end:end]
				offset = end
			}
		}
	}
	return sites
}

// alignShared aligns already shared [site-1][gene] inputs. All non-anchor
// products use one site-major batch; anchor inputs are copied unchanged.
func alignShared(mpcObj *mpc.MPC, embeddings, sites [][]mpc_core.RMat) []mpc_core.RMat {
	if len(embeddings) != mpcObj.GetNParty()-1 || len(sites) != len(embeddings) {
		panic("alignShared: expected embeddings and inputs for every data site")
	}
	genes := len(embeddings[0])
	aligned := make([]mpc_core.RMat, genes)
	var left, right []mpc_core.RMat
	for site := range sites {
		if len(embeddings[site]) != genes || len(sites[site]) != genes {
			panic("alignShared: inconsistent gene count")
		}
		for g := 0; g < genes; g++ {
			if site == 0 {
				aligned[g] = sites[site][g].Copy()
			} else {
				left = append(left, embeddings[site][g].Transpose())
				right = append(right, sites[site][g])
			}
		}
	}
	if len(left) > 0 {
		products := mpcObj.SSMultMats(left, right)
		for site := 1; site < len(sites); site++ {
			for g := range aligned {
				aligned[g].Add(products[(site-1)*genes+g])
			}
		}
	}
	return aligned
}

// batchEmbeddings selects genes in batch order, preserving the site axis.
func batchEmbeddings(batch GeneBatch, embeddings [][]mpc_core.RMat) [][]mpc_core.RMat {
	selected := make([][]mpc_core.RMat, len(embeddings))
	for site := range embeddings {
		selected[site] = make([]mpc_core.RMat, len(batch.GeneIndices))
		for position, geneIndex := range batch.GeneIndices {
			selected[site][position] = embeddings[site][geneIndex]
		}
	}
	return selected
}
