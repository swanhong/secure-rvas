package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	securecrypto "github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

// mux returns whenFalse + condition·(whenTrue − whenFalse). condition holds unscaled 0/1 bits (the
// comparison outputs and their products), so the product keeps the scale of the values and needs
// no truncation.
func mux(
	mpcObj *mpc.MPC,
	condition, whenTrue, whenFalse mpc_core.RVec,
) mpc_core.RVec {
	difference := whenTrue.Copy()
	difference.Sub(whenFalse)

	result := whenFalse.Copy()
	result.Add(mpcObj.SSMultElemVec(condition, difference))
	return result
}

func ComputeWeights(
	mpcObj *mpc.MPC,
	heParams *securecrypto.CryptoParams,
	dataParams DataParams,
	localDosage *mat.Dense,
	embeddings [][]mpc_core.RMat,
) (weight, signedWeight []mpc_core.RVec) {
	total := 0
	for _, gene := range dataParams.Genes {
		total += gene.Cap
	}
	weight = make([]mpc_core.RVec, len(dataParams.Genes))
	signedWeight = make([]mpc_core.RVec, len(dataParams.Genes))
	if total == 0 {
		for geneIndex := range dataParams.Genes {
			weight[geneIndex] = mpc_core.RVec{}
			signedWeight[geneIndex] = mpc_core.RVec{}
		}
		return weight, signedWeight
	}

	// count = Σk E_kᵀ c_k per gene, in the common coordinates.
	localCounts := make([]*mat.Dense, len(dataParams.Genes))
	if localDosage != nil {
		offset := 0
		for geneIndex, gene := range dataParams.Genes {
			localCounts[geneIndex] = mat.NewDense(gene.Cap, 1, localDosage.RawRowView(0)[offset:offset+gene.Cap])
			offset += gene.Cap
		}
	}
	var count mpc_core.RVec
	for _, aligned := range AlignShares(mpcObj, embeddings, localCounts, 1) {
		for _, row := range aligned {
			count = append(count, row[0])
		}
	}
	rtype := mpcObj.GetRType()
	fracBits := mpcObj.GetFracBits()

	// altMajor = [count > N]; counts are integers, so the threshold N + 1/2 makes a tie count as minor,
	// as in the plain reference.
	altMajor := mpcObj.NotLessThanPublic(
		count,
		rtype.FromFloat64(float64(dataParams.N)+0.5, fracBits),
		mpcObj.GetBooleanShareFlag(),
	)

	// complement = 2N - count
	totalAlleles := rtype.FromFloat64(float64(2*dataParams.N), fracBits)
	complement := mpc_core.InitRVec(rtype.Zero(), total)
	complement.Sub(count)
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		complement.AddScalar(totalAlleles)
	}
	base := mux(mpcObj, altMajor, count, complement)
	base.MulScalar(rtype.FromFloat64(1/float64(2*dataParams.N), fracBits))
	base = mpcObj.TruncVec(base, mpcObj.GetDataBits(), fracBits)

	// weight <- 25 * base^24
	baseCipher := mpcObj.SSToCVec(heParams, base)
	var base8 securecrypto.CipherVector
	if mpcObj.GetPid() != auxiliaryPartyID {
		base2 := securecrypto.CMult(heParams, baseCipher, baseCipher)
		base4 := securecrypto.CMult(heParams, base2, base2)
		base8 = securecrypto.CMult(heParams, base4, base4)
	}

	base8 = mpcObj.Network.CollectiveBootstrapVec(heParams, base8, auxiliaryPartyID)

	var base24 securecrypto.CipherVector
	if mpcObj.GetPid() != auxiliaryPartyID {
		base16 := securecrypto.CMult(heParams, base8, base8)
		base24 = securecrypto.CMult(heParams, base16, base8)
	}

	numCiphertexts := (total + heParams.GetSlots() - 1) / heParams.GetSlots()
	flatWeight := mpcObj.CVecToSS(
		heParams,
		rtype,
		base24,
		auxiliaryPartyID,
		numCiphertexts,
		total,
	)
	flatWeight.MulScalar(rtype.FromInt(25))

	negativeWeight := mpc_core.InitRVec(rtype.Zero(), len(flatWeight))
	negativeWeight.Sub(flatWeight)

	flatSignedWeight := mux(mpcObj, altMajor, negativeWeight, flatWeight)

	offset := 0
	for geneIndex, gene := range dataParams.Genes {
		end := offset + gene.Cap
		weight[geneIndex] = flatWeight[offset:end].Copy()
		signedWeight[geneIndex] = flatSignedWeight[offset:end].Copy()
		offset = end
	}

	return weight, signedWeight
}
