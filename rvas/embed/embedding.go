package embed

import (
	"math/big"
	"math/bits"

	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/crypto"
	"github.com/hhcho/sfgwas/mpc"
)

// FinalizeEmbedding converts E1 coordinates into exact secret-shared embeddings.
// All parties, including P0, call it after AssignCoordinates. Each data site
// supplies only its own CoordinateResult. Output is
// [site-1][gene][row][column], shared across P1..PN; P0's shares are zero.
// Dummy rows are one-hot; their padded statistics must be zero.
// E1=I_M, including padded anchor rows.
//
// C2S keeps its existing magnitude admission bound (2^(dataBits-fracBits));
// inputs must fit that bound. Coordinates need total HE+refresh+C2S error
// strictly <1/2.
// There is no plaintext reveal, approximate bit gate, or randomized truncation.
func FinalizeEmbedding(cp *crypto.CryptoParams, m *mpc.MPC, p Setting, genes int, coordinates CoordinateResult) [][]mpc_core.RMat {
	if p.validate() != nil || genes < 0 || m.GetNParty() < 3 || m.GetFracBits() < 1 {
		panic("embedding: invalid setting/site count/fixed-point scale")
	}

	rtype := m.GetRType()
	out := make([][]mpc_core.RMat, m.GetNParty()-1)
	for site := range out {
		out[site] = make([]mpc_core.RMat, genes)
	}
	for g := 0; g < genes; g++ {
		out[0][g] = mpc_core.InitRMat(rtype.Zero(), p.M, p.M)
		if m.GetPid() == 1 {
			for j := 0; j < p.M; j++ {
				out[0][g][j][j] = rtype.One()
			}
		}
	}
	if genes == 0 {
		return out
	}
	for source := 2; source < m.GetNParty(); source++ {
		packed := packDestinations(cp, m.Network, source, p, genes, coordinates.Destinations)
		fixed := collectGeneShares(cp, m, p, genes, source, packed, p.M)
		decoded := recoverAndDecode(m, fixed, p.M)
		for g := 0; g < genes; g++ {
			if m.GetPid() == 0 {
				// MPC helpers return preprocessing scratch, not output shares.
				out[source-1][g] = mpc_core.InitRMat(rtype.Zero(), p.M, p.M)
			} else {
				out[source-1][g] = decoded[g*p.M : (g+1)*p.M]
			}
		}
	}
	return out
}

// Each row's coordinate is already repeated in its gene block. Selecting only
// gR+j and adding across rows gathers M contiguous destinations per gene, with
// R-M public padding slots between genes. No rotations are needed.
func packDestinations(cp *crypto.CryptoParams, net *mpc.Network, source int, p Setting, genes int, rows crypto.CipherMatrix) crypto.CipherVector {
	if net.GetPid() == 0 {
		return nil
	}
	layout := DenseShape{M: p.Rho + p.Sigma, Genes: genes}
	chunks := layout.chunks(cp)
	rows = net.RefreshCMatWithRandomZero(cp, rows, source, p.M, chunks)
	if net.GetPid() != source {
		return nil
	}
	out := make(crypto.CipherVector, chunks)
	for chunk := 0; chunk < chunks; chunk++ {
		for j := 0; j < p.M; j++ {
			mask := make([]complex128, cp.GetSlots())
			for g := 0; g < layout.genes(cp, chunk); g++ {
				mask[g*layout.M+j] = 1
			}
			x := crypto.MulComplexSlots(cp, rows[j][chunk], mask)
			out[chunk] = addCiphertext(cp, out[chunk], x)
		}
	}
	return out
}

// Convert each physical chunk separately: retain its R stride, then select
// public gR+j indices locally on shares. P0 skips C2S and joins the following MPC.
func collectGeneShares(cp *crypto.CryptoParams, m *mpc.MPC, p Setting, genes, source int, packed crypto.CipherVector, width int) mpc_core.RVec {
	out := mpc_core.InitRVec(m.GetRType().Zero(), genes*width)
	if m.GetPid() == 0 {
		return out
	}
	layout := DenseShape{M: p.Rho + p.Sigma, Genes: genes}
	perChunk := cp.GetSlots() / layout.M
	for chunk := 0; chunk < layout.chunks(cp); chunk++ {
		var input crypto.CipherVector
		if m.GetPid() == source {
			input = crypto.CipherVector{packed[chunk]}
		}
		count := layout.genes(cp, chunk)
		shares := m.CVecToSS(cp, m.GetRType(), input, source, 1, (count-1)*layout.M+width)
		for g := 0; g < count; g++ {
			copy(out[(chunk*perChunk+g)*width:], shares[g*layout.M:g*layout.M+width])
		}
	}
	return out
}

// Recover and Decode are fused: extract nearest-integer bits, then run the
// existing binary decoder. The separate residual preserves the input shares.
func recoverAndDecode(m *mpc.MPC, fixed mpc_core.RVec, capM int) mpc_core.RMat {
	frac, rtype := m.GetFracBits(), m.GetRType()
	if capM < 1 || capM&(capM-1) != 0 || frac < 1 || frac+bits.Len(uint(capM))+1 >= rtype.ModBitLength() {
		panic("Recover: invalid cap or fixed-point range")
	}
	residual := fixed.Copy()
	indexBits := make([]mpc_core.RVec, bits.Len(uint(capM))-1)
	half := new(big.Int).Lsh(big.NewInt(1), uint(frac-1))
	for b := len(indexBits) - 1; b >= 0; b-- {
		weight := new(big.Int).Lsh(big.NewInt(1), uint(frac+b))
		threshold := rtype.FromBigInt(new(big.Int).Sub(weight, half))
		x := residual.Copy()
		if m.GetPid() == 1 {
			for j := range x {
				x[j] = x[j].Sub(threshold)
			}
		}
		indexBits[b] = m.IsPositive(x, false)
		scaled := rtype.FromBigInt(weight)
		for j := range residual {
			residual[j] = residual[j].Sub(indexBits[b][j].Mul(scaled))
		}
	}
	return m.DecodeBits(len(fixed), indexBits)
}
