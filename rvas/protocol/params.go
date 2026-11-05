package protocol

import "fmt"

type Gene struct {
	GeneID string
	Cap    int // public cap M_g from the alignment, a power of two; every gene matrix has Cap rows
}

type DataParams struct {
	Genes          []Gene
	N              int
	C              int
	PhenotypeCount int
}

type GeneBatch struct {
	W           int
	GeneIndices []int
}

type CryptoParams struct {
	Slots   int
	R       int
	Batches []GeneBatch
}

func BuildCryptoParams(dataParams DataParams, probeCount int, slots int) (CryptoParams, error) {
	if slots < 1 {
		return CryptoParams{}, fmt.Errorf("slots must be a positive integer")
	}

	if probeCount < 1 {
		return CryptoParams{}, fmt.Errorf("probeCount must be a positive integer")
	}

	genesByWidth := make(map[int][]int)
	maxWidth := 1

	// The packing width of a gene is its cap: a power of two, at most the slot count.
	for geneIndex, gene := range dataParams.Genes {
		if gene.Cap < 1 || gene.Cap&(gene.Cap-1) != 0 || gene.Cap > slots {
			return CryptoParams{}, fmt.Errorf("gene %s has cap %d; want a power of two at most the slot count %d", gene.GeneID, gene.Cap, slots)
		}
		genesByWidth[gene.Cap] = append(genesByWidth[gene.Cap], geneIndex)
		maxWidth = max(maxWidth, gene.Cap)
	}

	cryptoParams := CryptoParams{
		Slots:   slots,
		R:       probeCount,
		Batches: []GeneBatch{},
	}

	for width := 1; width <= maxWidth; width *= 2 {
		geneIndices := genesByWidth[width]
		if len(geneIndices) == 0 {
			continue
		}
		genesPerBatch := slots / width
		for start := 0; start < len(geneIndices); start += genesPerBatch {
			end := min(start+genesPerBatch, len(geneIndices))
			batchIndices := append([]int(nil), geneIndices[start:end]...)
			cryptoParams.Batches = append(cryptoParams.Batches, GeneBatch{
				W:           width,
				GeneIndices: batchIndices,
			})
		}
	}

	return cryptoParams, nil
}
