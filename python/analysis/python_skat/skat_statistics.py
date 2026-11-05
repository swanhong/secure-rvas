import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiuParameters:
    mean: float
    standard_deviation: float
    scale: float
    degrees: float
    ncp: float


@dataclass(frozen=True)
class GeneStatistics:
    skat: np.ndarray
    eigenvalues: np.ndarray
    liu: LiuParameters
    modified_liu: LiuParameters
    burden: np.ndarray
    burden_liu: LiuParameters


def fit_null_models(
    covariates: np.ndarray,
    phenotypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(covariates.shape[0]), covariates))
    basis, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = (
        max(design.shape)
        * np.finfo(np.float64).eps
        * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    basis = basis[:, :rank]
    residuals = phenotypes - basis @ (basis.T @ phenotypes)
    residual_variances = np.sum(residuals * residuals, axis=0) / (
        design.shape[0] - rank
    )
    return basis, residuals, residual_variances


def prepare_weighted_genotype(genotype: np.ndarray) -> np.ndarray:
    genotype = genotype.copy()
    keep = np.std(genotype, axis=0, ddof=1) > 0
    genotype = genotype[:, keep]
    allele_frequencies = np.mean(genotype, axis=0) / 2
    flip = allele_frequencies > 0.5
    genotype[:, flip] = 2 - genotype[:, flip]
    allele_frequencies[flip] = 1 - allele_frequencies[flip]

    weights = 25 * np.power(1 - allele_frequencies, 24)
    return genotype * weights


def regularized_gamma_q(shape: float, value: float) -> float:
    if value <= 0:
        return 1.0

    log_factor = -value + shape * math.log(value) - math.lgamma(shape)
    epsilon = 1e-15
    if value < shape + 1:
        term = 1 / shape
        total = term
        denominator = shape
        for _ in range(10000):
            denominator += 1
            term *= value / denominator
            total += term
            if abs(term) <= abs(total) * epsilon:
                break
        return 1 - total * math.exp(log_factor)

    minimum = 1e-300
    denominator = value + 1 - shape
    continued_fraction_c = 1 / minimum
    continued_fraction_d = 1 / max(abs(denominator), minimum)
    continued_fraction = continued_fraction_d
    for index in range(1, 10001):
        coefficient = -index * (index - shape)
        denominator += 2
        continued_fraction_d = denominator + coefficient * continued_fraction_d
        if abs(continued_fraction_d) < minimum:
            continued_fraction_d = minimum
        continued_fraction_c = denominator + coefficient / continued_fraction_c
        if abs(continued_fraction_c) < minimum:
            continued_fraction_c = minimum
        continued_fraction_d = 1 / continued_fraction_d
        change = continued_fraction_d * continued_fraction_c
        continued_fraction *= change
        if abs(change - 1) <= epsilon:
            break
    return math.exp(log_factor) * continued_fraction


def noncentral_chi_square_sf(value: float, degrees: float, ncp: float) -> float:
    if value <= 0:
        return 1.0
    if ncp <= 1e-14:
        return regularized_gamma_q(degrees / 2, value / 2)

    poisson_mean = ncp / 2
    center = int(math.floor(poisson_mean))
    center_weight = math.exp(
        -poisson_mean
        + center * math.log(poisson_mean)
        - math.lgamma(center + 1)
    )
    total = center_weight * regularized_gamma_q(
        degrees / 2 + center,
        value / 2,
    )

    weight = center_weight
    for index in range(center, 0, -1):
        weight *= index / poisson_mean
        total += weight * regularized_gamma_q(
            degrees / 2 + index - 1,
            value / 2,
        )

    weight = center_weight
    for index in range(center + 1, center + 100000):
        weight *= poisson_mean / index
        total += weight * regularized_gamma_q(
            degrees / 2 + index,
            value / 2,
        )
        if weight < 1e-16:
            break
    return min(1.0, max(0.0, total))


def liu_parameters(eigenvalues: np.ndarray, modified: bool) -> LiuParameters:
    powers = [
        float(np.sum(np.power(eigenvalues, power)))
        for power in range(1, 5)
    ]
    standard_deviation = math.sqrt(2 * powers[1])
    skewness = powers[2] / math.pow(powers[1], 1.5)
    kurtosis = powers[3] / math.pow(powers[1], 2)

    if skewness * skewness > kurtosis:
        scale = 1 / (
            skewness - math.sqrt(skewness * skewness - kurtosis)
        )
        ncp = skewness * math.pow(scale, 3) - scale * scale
        degrees = scale * scale - 2 * ncp
    elif modified:
        degrees = 1 / kurtosis
        scale = math.sqrt(degrees)
        ncp = 0.0
    else:
        scale = 1 / skewness
        degrees = 1 / (skewness * skewness)
        ncp = 0.0

    return LiuParameters(
        mean=powers[0],
        standard_deviation=standard_deviation,
        scale=scale,
        degrees=degrees,
        ncp=ncp,
    )


def liu_p_value(statistic: float, parameters: LiuParameters) -> float:
    transformed = (
        (statistic - parameters.mean)
        / parameters.standard_deviation
        * (math.sqrt(2) * parameters.scale)
        + parameters.degrees
        + parameters.ncp
    )
    return noncentral_chi_square_sf(
        transformed,
        parameters.degrees,
        parameters.ncp,
    )


def skat_eigenvalues(
    weighted_genotype: np.ndarray,
    covariate_basis: np.ndarray,
) -> np.ndarray:
    residualized = weighted_genotype - covariate_basis @ (
        covariate_basis.T @ weighted_genotype
    )
    kernel = weighted_genotype.T @ residualized / 2
    raw = np.linalg.eigvalsh(kernel)
    nonnegative = raw[raw >= 0]
    threshold = float(np.mean(nonnegative)) / 1e5
    return raw[raw > threshold]


def compute_gene_statistics(
    weighted_genotype: np.ndarray,
    covariate_basis: np.ndarray,
    residuals: np.ndarray,
    residual_variances: np.ndarray,
) -> GeneStatistics:
    scores = residuals.T @ weighted_genotype
    skat = np.sum(scores * scores, axis=1) / (2 * residual_variances)
    eigenvalues = skat_eigenvalues(weighted_genotype, covariate_basis)

    burden_genotype = np.sum(weighted_genotype, axis=1, keepdims=True)
    burden_scores = residuals.T @ burden_genotype
    burden = np.square(burden_scores[:, 0]) / (2 * residual_variances)
    burden_eigenvalue = skat_eigenvalues(
        burden_genotype,
        covariate_basis,
    )

    return GeneStatistics(
        skat=skat,
        eigenvalues=eigenvalues,
        liu=liu_parameters(eigenvalues, modified=False),
        modified_liu=liu_parameters(eigenvalues, modified=True),
        burden=burden,
        burden_liu=liu_parameters(burden_eigenvalue, modified=True),
    )
