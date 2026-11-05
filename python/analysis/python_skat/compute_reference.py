#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

from davies_qfc import Davies
from load_preprocessed import PlainInput, read_gene_genotype, read_plain_input, read_site_inputs
from skat_statistics import (
    compute_gene_statistics,
    fit_null_models,
    liu_p_value,
    prepare_weighted_genotype,
)


OUTPUT_COLUMNS = [
    "gene_index",
    "gene_id",
    "phenotype_index",
    "burden_p",
    "skat_davies_p",
    "skat_davies_converged",
    "skat_liu_p",
]


def analyze(input_data: PlainInput) -> list[dict[str, str | int | float]]:
    covariate_basis, residuals, residual_variances = fit_null_models(
        input_data.covariates,
        input_data.phenotypes,
    )
    davies = Davies()
    rows: list[dict[str, str | int | float]] = []

    for gene_index, gene_id in enumerate(input_data.genes):
        genotype = read_gene_genotype(input_data, gene_index)
        weighted = prepare_weighted_genotype(genotype)
        if weighted.shape[1] == 0:
            for phenotype_index in range(input_data.phenotypes.shape[1]):
                rows.append(
                    {
                        "gene_index": gene_index,
                        "gene_id": gene_id,
                        "phenotype_index": phenotype_index,
                        "burden_p": "NA",
                        "skat_davies_p": "NA",
                        "skat_davies_converged": "NA",
                        "skat_liu_p": "NA",
                    }
                )
            continue

        statistics = compute_gene_statistics(
            weighted,
            covariate_basis,
            residuals,
            residual_variances,
        )
        for phenotype_index, statistic in enumerate(statistics.skat):
            burden_p = liu_p_value(
                float(statistics.burden[phenotype_index]),
                statistics.burden_liu,
            )
            liu_p = liu_p_value(float(statistic), statistics.liu)
            modified_liu_p = liu_p_value(
                float(statistic),
                statistics.modified_liu,
            )
            davies_p, converged = davies.p_value(
                float(statistic),
                statistics.eigenvalues,
                modified_liu_p,
            )
            rows.append(
                {
                    "gene_index": gene_index,
                    "gene_id": gene_id,
                    "phenotype_index": phenotype_index,
                    "burden_p": burden_p,
                    "skat_davies_p": davies_p,
                    "skat_davies_converged": converged,
                    "skat_liu_p": liu_p,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="*")
    parser.add_argument("--inputs-json", help="ordered P1..PN configured input paths")
    args = parser.parse_args()
    if args.inputs_json is not None:
        if args.directories:
            parser.error("use --inputs-json or legacy prepared directories, not both")
        input_data = read_site_inputs(json.loads(args.inputs_json))
    else:
        if len(args.directories) not in (1, 2):
            parser.error("provide one or two legacy prepared directories, or --inputs-json")
        input_data = read_plain_input(*map(Path, args.directories))
    rows = analyze(input_data)
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=OUTPUT_COLUMNS,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
