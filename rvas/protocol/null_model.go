package protocol

import (
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
	"gonum.org/v1/gonum/mat"
)

func SetupNull(
	mpcObj *mpc.MPC,
	dataParams DataParams,
	x *mat.Dense,
	y0 *mat.Dense,
	observe func(stage string) func(),
) (
	beta mpc_core.RMat,
	xtxInv mpc_core.RMat,
	rss mpc_core.RVec,
) {
	/*
		beta = (XᵀX)⁻¹XᵀY by a secure Cholesky factorization of XᵀX, xtxInv = (XᵀX)⁻¹, and
		rss[t] = yᵀy − 2βᵀXᵀy + βᵀXᵀXβ for phenotype t. XᵀX must be positive definite and every
		rss[t] positive.
	*/
	covariateCount := dataParams.C
	phenotypeCount := dataParams.PhenotypeCount

	var localXtx *mat.Dense
	var localXty *mat.Dense
	var localYty *mat.Dense

	done := observe("null_local_equations")
	if mpcObj.GetPid() != auxiliaryPartyID {
		localXtx = new(mat.Dense)
		localXtx.Mul(x.T(), x)

		localXty = new(mat.Dense)
		localXty.Mul(x.T(), y0)

		localYty = mat.NewDense(1, phenotypeCount, nil)
		for pheno := 0; pheno < phenotypeCount; pheno++ {
			column := y0.ColView(pheno)
			localYty.Set(0, pheno, mat.Dot(column, column))
		}
	}
	done()

	done = observe("null_aggregate_shares")
	xtx := ShareSum(mpcObj, localXtx, covariateCount, covariateCount)
	xty := ShareSum(mpcObj, localXty, covariateCount, phenotypeCount)
	yty := ShareSum(mpcObj, localYty, 1, phenotypeCount)[0]
	done()

	done = observe("null_factor_solve")
	lower, diagInv := FactorSPD(mpcObj, xtx)
	beta = SolveSPD(mpcObj, lower, diagInv, xty)

	rtype := mpcObj.GetRType()
	identity := mpc_core.InitRMat(rtype.Zero(), covariateCount, covariateCount)
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		one := rtype.FromFloat64(1.0, mpcObj.GetFracBits())
		for diag := 0; diag < covariateCount; diag++ {
			identity[diag][diag] = one.Copy()
		}
	}
	xtxInv = SolveSPD(mpcObj, lower, diagInv, identity)
	done()

	done = observe("null_rss")
	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()

	xtxBeta := mpcObj.SSMultMat(xtx, beta)
	xtxBeta = mpcObj.TruncMat(xtxBeta, dataBits, fracBits)

	xtyBeta := mpcObj.SSMultElemMat(xty, beta)
	xtyBeta = mpcObj.TruncMat(xtyBeta, dataBits, fracBits)

	betaXtxBeta := mpcObj.SSMultElemMat(beta, xtxBeta)
	betaXtxBeta = mpcObj.TruncMat(betaXtxBeta, dataBits, fracBits)

	rss = mpc_core.InitRVec(rtype.Zero(), phenotypeCount)
	for pheno := 0; pheno < phenotypeCount; pheno++ {
		value := yty[pheno].Copy()
		for cov := 0; cov < covariateCount; cov++ {
			value = value.Sub(xtyBeta[cov][pheno])
			value = value.Sub(xtyBeta[cov][pheno])
			value = value.Add(betaXtxBeta[cov][pheno])
		}
		rss[pheno] = value
	}
	done()
	return beta, xtxInv, rss
}
