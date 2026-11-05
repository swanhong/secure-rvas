package protocol

import mpc_core "github.com/hhcho/mpc-core"

func PackGeneBatch[T any](
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	values [][]T,
) []T {
	nu := cryptoParams.Slots / batch.W
	h := nu / len(batch.GeneIndices)
	packed := make([]T, cryptoParams.Slots)

	for pos, geneIndex := range batch.GeneIndices {
		laneBase := pos * h
		capM := dataParams.Genes[geneIndex].Cap
		for variantIndex := 0; variantIndex < capM; variantIndex++ {
			slot := variantIndex*nu + laneBase
			packed[slot] = values[pos][variantIndex]
		}
	}
	return packed
}

func packSharedGeneBatch(
	rtype mpc_core.RElem,
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	values [][]mpc_core.RElem,
) mpc_core.RVec {
	packed := mpc_core.RVec(PackGeneBatch(
		dataParams,
		cryptoParams,
		batch,
		values,
	))
	for slot, value := range packed {
		if value == nil {
			packed[slot] = rtype.Zero()
		}
	}
	return packed
}

func UnpackGeneBatch[T any](
	dataParams DataParams,
	cryptoParams CryptoParams,
	batch GeneBatch,
	packed []T,
) [][]T {
	nu := cryptoParams.Slots / batch.W
	h := nu / len(batch.GeneIndices)
	values := make([][]T, len(batch.GeneIndices))

	for pos, geneIndex := range batch.GeneIndices {
		laneBase := pos * h
		capM := dataParams.Genes[geneIndex].Cap
		values[pos] = make([]T, capM)
		for variantIndex := 0; variantIndex < capM; variantIndex++ {
			slot := variantIndex*nu + laneBase
			values[pos][variantIndex] = packed[slot]
		}
	}
	return values
}
