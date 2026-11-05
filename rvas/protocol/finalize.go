package protocol

import (
	"math"
	"sync"

	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/mpc"
)

func Finalize(
	mpcObjects []*mpc.MPC,
	dataParams DataParams,
	geneQ, geneL mpc_core.RMat,
	rss, geneV, geneInvS1, geneS2, geneS3 mpc_core.RVec,
	observe func(stage string) func(),
) (
	bSquared, z mpc_core.RVec,
) {
	/*
		For every gene g and phenotype t:
		Q and V are divided by N; L is divided by sqrt(N).
		invS1, S2, and S3 are 1/S1, S2/S1², and S3/S1³ for K/(2*N),
		where K = W * (GᵀG - GᵀX*(XᵀX)^-1*XᵀG) * W.

		alpha[t] = (N - C) / (2 * rss[t])

		Qs[g,t] = alpha[t] * geneQ[g,t] / S1[g]
		Qb[g,t] = alpha[t] * geneL[g,t]^2

		b²[g,t] = Qb[g,t] / V[g] = alpha[t] * (geneL[g,t] / sqrt(V[g]))²,
		          the square of the Burden pivot b = |Z_B|/√2
		z[g,t]  = WilsonHilferty(Qs[g,t], S2[g]/S1[g]², S3[g]/S1[g]³), Qs having null mean 1

		Outputs use gene-major order g*q+t.
	*/
	mpcObj := mpcObjects[0]
	rtype := mpcObj.GetRType()
	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()
	geneCount := len(dataParams.Genes)
	phenotypeCount := dataParams.PhenotypeCount
	outputLength := geneCount * phenotypeCount

	multiply := func(
		workerMPC *mpc.MPC,
		left, right mpc_core.RVec,
	) mpc_core.RVec {
		product := workerMPC.SSMultElemVec(left, right)
		return workerMPC.TruncVec(product, dataBits, fracBits)
	}

	runChunks := func(
		work func(workerMPC *mpc.MPC, start, end int),
	) {
		workerCount := len(mpcObjects)
		if workerCount > geneCount {
			workerCount = geneCount
		}

		var workers sync.WaitGroup
		workers.Add(workerCount)

		for lane := 0; lane < workerCount; lane++ {
			start := lane * geneCount / workerCount
			end := (lane + 1) * geneCount / workerCount

			go func(lane, start, end int) {
				defer workers.Done()
				work(mpcObjects[lane], start, end)
			}(lane, start, end)
		}

		workers.Wait()
	}

	// 1. Compute alpha[t] = (N - C) / (2 * rss[t]).
	done := observe("alpha_score_assembly")
	alphaNumerator := mpc_core.InitRVec(rtype.Zero(), phenotypeCount)
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		alphaNumerator.AddScalar(
			rtype.FromFloat64(
				0.5*float64(dataParams.N-dataParams.C),
				fracBits,
			),
		)
	}
	alpha := mpcObj.Divide(alphaNumerator, rss, false)

	// 2. Assemble gene-phenotype values in g*q+t order.
	alphaByGene := mpc_core.InitRVec(rtype.Zero(), outputLength)
	scoreQuadratic := mpc_core.InitRVec(rtype.Zero(), outputLength)
	burdenLinear := mpc_core.InitRVec(rtype.Zero(), outputLength)
	invS1 := mpc_core.InitRVec(rtype.Zero(), outputLength)

	for gene := 0; gene < geneCount; gene++ {
		for phenotype := 0; phenotype < phenotypeCount; phenotype++ {
			index := gene*phenotypeCount + phenotype

			alphaByGene[index] = alpha[phenotype].Copy()

			scoreQuadratic[index] = geneQ[gene][phenotype].Copy()
			burdenLinear[index] = geneL[gene][phenotype].Copy()

			invS1[index] = geneInvS1[gene].Copy()
		}
	}

	// 3. Compute Qs.
	scoreQuadratic = multiply(mpcObj, scoreQuadratic, invS1)
	scoreQuadratic = multiply(mpcObj, alphaByGene, scoreQuadratic)
	done()

	// 4. Compute b² = Qb / V = (alpha * u) * u with u = L / sqrt(V). L is signed, so no square root of
	// Qb is taken; 1/sqrt(V) keeps more precision for a large V than 1/V would; and multiplying by
	// alpha before the second u keeps a small u² from underflowing when alpha is large.
	done = observe("burden_statistic")
	bSquared = mpc_core.InitRVec(rtype.Zero(), outputLength)

	runChunks(func(workerMPC *mpc.MPC, start, end int) {
		_, invSqrtVariance := workerMPC.SqrtAndSqrtInverse(geneV[start:end], false)
		lanes := burdenLinear[start*phenotypeCount : end*phenotypeCount]
		standardized := multiply(workerMPC, lanes, repeatGeneValues(invSqrtVariance, phenotypeCount))
		scaled := multiply(workerMPC, alphaByGene[start*phenotypeCount:end*phenotypeCount], standardized)
		copy(bSquared[start*phenotypeCount:end*phenotypeCount], multiply(workerMPC, scaled, standardized))
	})
	done()

	// 5. Compute the Wilson-Hilferty SKAT pivot.
	done = observe("skat_wilson_hilferty")
	z = mpc_core.InitRVec(rtype.Zero(), outputLength)

	runChunks(func(workerMPC *mpc.MPC, start, end int) {
		copy(z[start*phenotypeCount:end*phenotypeCount], wilsonHilferty(
			workerMPC,
			scoreQuadratic[start*phenotypeCount:end*phenotypeCount],
			geneS2[start:end],
			geneS3[start:end],
			phenotypeCount,
		))
	})
	done()
	return bSquared, z
}

// repeatGeneValues expands gene shares into the existing gene-major phenotype order.
func repeatGeneValues(values mpc_core.RVec, phenotypeCount int) mpc_core.RVec {
	result := make(mpc_core.RVec, len(values)*phenotypeCount)
	for gene, value := range values {
		for phenotype := 0; phenotype < phenotypeCount; phenotype++ {
			result[gene*phenotypeCount+phenotype] = value.Copy()
		}
	}
	return result
}

func wilsonHilferty(
	mpcObj *mpc.MPC,
	qStatistic, s2, s3 mpc_core.RVec,
	phenotypeCount int,
) mpc_core.RVec {
	/*
		s2 and s3 contain one value per gene; qStatistic uses gene-major phenotype order and is
		normalized to null mean 1 (q* = 2Q/S1), with S2 and S3 the matching normalized moments.

		Compute the Wilson-Hilferty pivot:

		gamma = S3 / S2^(3/2)
		eta   = 2 * gamma^2 / 9

		z =
		    (cbrt(1 + (Q - 1) * gamma / sqrt(S2))
		        - 1 + eta)
		    / sqrt(eta)

		S2 and eta must be positive.
	*/

	rtype := mpcObj.GetRType()
	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()

	multiply := func(left, right mpc_core.RVec) mpc_core.RVec {
		product := mpcObj.SSMultElemVec(left, right)
		return mpcObj.TruncVec(product, dataBits, fracBits)
	}

	// 1. Compute gamma = S3 / S2^(3/2).
	_, invSqrtS2 := mpcObj.SqrtAndSqrtInverse(s2, false)

	gamma := multiply(s3, invSqrtS2)
	gamma = multiply(gamma, invSqrtS2)
	gamma = multiply(gamma, invSqrtS2)

	// These values depend only on the gene, so compute them before expanding phenotypes.
	// 2. Compute eta = 2 * gamma^2 / 9.
	eta := multiply(gamma, gamma)
	eta.MulScalar(rtype.FromFloat64(2.0/9.0, fracBits))
	eta = mpcObj.TruncVec(eta, dataBits, fracBits)

	_, invSqrtEta := mpcObj.SqrtAndSqrtInverse(eta, false)
	for _, values := range []*mpc_core.RVec{&invSqrtS2, &gamma, &eta, &invSqrtEta} {
		*values = repeatGeneValues(*values, phenotypeCount)
	}
	// 3. Compute argument = 1 + (Q - 1) * gamma / sqrt(S2).
	qMinusOne := qStatistic.Copy()
	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		qMinusOne.AddScalar(rtype.FromFloat64(-1, fracBits))
	}

	argument := multiply(qMinusOne, invSqrtS2)
	argument = multiply(argument, gamma)

	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		argument.AddScalar(rtype.FromFloat64(1, fracBits))
	}

	// 4. Compute z = (cbrt(argument) - 1 + eta) / sqrt(eta).
	root := SecureCubeRoot(mpcObj, argument)
	numerator := root.Copy()
	numerator.Add(eta)

	if mpcObj.GetPid() == mpcObj.GetHubPid() {
		numerator.AddScalar(
			rtype.FromFloat64(-1, fracBits),
		)
	}

	// 5. z = numerator / sqrt(eta).
	return multiply(numerator, invSqrtEta)
}

func SecureCubeRoot(
	mpcObj *mpc.MPC,
	values mpc_core.RVec,
) mpc_core.RVec {
	/*
		Compute the real signed cube root of shared fixed-point values.

		1. Separate sign, zero, and magnitude.
		2. Reduce large magnitudes using powers of 8.
		3. Raise small nonzero magnitudes using powers of 8.
		4. Compute the cube root on [0.1, 8) using Newton iteration.
		5. Restore the original range and sign.

		y <- (4y - x*y^4) / 3
		cbrt(x) = x*y^2
	*/

	if len(values) == 0 {
		return mpc_core.RVec{}
	}

	rtype := mpcObj.GetRType()
	dataBits := mpcObj.GetDataBits()
	fracBits := mpcObj.GetFracBits()
	booleanShares := mpcObj.GetBooleanShareFlag()
	length := len(values)

	publicVector := func(value float64) mpc_core.RVec {
		result := mpc_core.InitRVec(rtype.Zero(), length)
		if mpcObj.GetPid() == mpcObj.GetHubPid() {
			result.AddScalar(rtype.FromFloat64(value, fracBits))
		}
		return result
	}

	multiply := func(left, right mpc_core.RVec) mpc_core.RVec {
		product := mpcObj.SSMultElemVec(left, right)
		return mpcObj.TruncVec(product, dataBits, fracBits)
	}

	square := func(value mpc_core.RVec) mpc_core.RVec {
		return multiply(value, value)
	}

	scale := func(value mpc_core.RVec, factor float64) mpc_core.RVec {
		result := value.Copy()
		result.MulScalar(rtype.FromFloat64(factor, fracBits))
		return mpcObj.TruncVec(result, dataBits, fracBits)
	}

	addScaledBits := func(
		result mpc_core.RVec,
		bits mpc_core.RVec,
		coefficient float64,
	) {
		encodedCoefficient := rtype.FromFloat64(coefficient, fracBits)
		for index := range result {
			result[index] = result[index].Add(
				bits[index].Mul(encodedCoefficient),
			)
		}
	}

	integerBits := dataBits - fracBits - 1
	if integerBits < 4 || fracBits < 3 {
		panic(
			"SecureCubeRoot requires at least 4 integer bits and 3 fractional bits",
		)
	}

	positiveCeil := func(value float64) int {
		if value <= 0 {
			return 0
		}
		return int(math.Ceil(value))
	}

	// 1. Separate sign, zero, and magnitude.
	negative := mpcObj.LessThanPublic(
		values,
		rtype.Zero(),
		booleanShares,
	)

	minimumPositive := rtype.FromFloat64(
		math.Ldexp(1, -fracBits),
		fracBits,
	)
	positive := mpcObj.NotLessThanPublic(
		values,
		minimumPositive,
		booleanShares,
	)

	nonzero := negative.Copy()
	nonzero.Add(positive)
	zero := mpcObj.FlipBit(nonzero)

	negated := values.Copy()
	for index := range negated {
		negated[index] = negated[index].Neg()
	}
	magnitude := mux(mpcObj, negative, negated, values)

	rootScales := make([]mpc_core.RVec, 0)

	// 2. Reduce large magnitudes into the Newton interval.
	highSteps := positiveCeil(
		(float64(integerBits) - math.Log2(9.0)) / 3.0,
	)
	for remaining := highSteps; remaining > 0; {
		groupSize := remaining
		if groupSize > fracBits/3 {
			groupSize = fracBits / 3
		}

		rangeScale := publicVector(1)
		rootScale := publicVector(1)
		threshold := 8.0
		rootCoefficient := 1.0

		for step := 0; step < groupSize; step++ {
			selected := mpcObj.NotLessThanPublic(
				magnitude,
				rtype.FromFloat64(threshold, fracBits),
				booleanShares,
			)

			addScaledBits(
				rangeScale,
				selected,
				-7.0/threshold,
			)
			addScaledBits(
				rootScale,
				selected,
				rootCoefficient,
			)

			threshold *= 8
			rootCoefficient *= 2
		}

		magnitude = multiply(magnitude, rangeScale)
		rootScales = append(rootScales, rootScale)
		remaining -= groupSize
	}

	// 3. Raise small nonzero magnitudes into the Newton interval.
	lowSteps := positiveCeil(
		(float64(fracBits) + math.Log2(0.1)) / 3.0,
	)
	for remaining := lowSteps; remaining > 0; {
		groupSize := remaining
		if groupSize > (integerBits-1)/3 {
			groupSize = (integerBits - 1) / 3
		}

		rangeScale := publicVector(1)
		rootScale := publicVector(1)
		threshold := 0.1
		rangeCoefficient := 7.0
		rootCoefficient := 0.5

		for step := 0; step < groupSize; step++ {
			selected := mpcObj.LessThanPublic(
				magnitude,
				rtype.FromFloat64(threshold, fracBits),
				booleanShares,
			)

			addScaledBits(
				rangeScale,
				selected,
				rangeCoefficient,
			)
			addScaledBits(
				rootScale,
				selected,
				-rootCoefficient,
			)

			threshold /= 8
			rangeCoefficient *= 8
			rootCoefficient *= 0.5
		}

		magnitude = multiply(magnitude, rangeScale)
		rootScales = append(rootScales, rootScale)
		remaining -= groupSize
	}

	// Use a safe positive input if range reduction leaves a tiny value.
	tiny := mpcObj.LessThanPublic(
		magnitude,
		rtype.FromFloat64(0.1, fracBits),
		booleanShares,
	)
	reduced := mux(
		mpcObj,
		tiny,
		publicVector(0.1),
		magnitude,
	)

	// 4. Compute cbrt(reduced) on [0.1, 8).
	inverseRoot := publicVector(0.7)
	for iteration := 0; iteration < 8; iteration++ {
		inverseRootSquared := square(inverseRoot)
		inverseRootFourth := square(inverseRootSquared)
		valueTimesFourth := multiply(
			reduced,
			inverseRootFourth,
		)

		next := inverseRoot.Copy()
		next.MulScalar(rtype.FromInt(4))
		next.Sub(valueTimesFourth)

		inverseRoot = scale(next, 1.0/3.0)
	}

	root := multiply(
		reduced,
		square(inverseRoot),
	)

	// 5. Restore the original range and sign.
	for _, rootScale := range rootScales {
		root = multiply(root, rootScale)
	}

	sign := publicVector(1)
	addScaledBits(sign, negative, -2)
	addScaledBits(sign, zero, -1)

	return multiply(root, sign)
}
