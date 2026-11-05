#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

from config import load_party_config


KEY_COLUMNS = [
    "chromosome",
    "gene_index",
    "gene_id",
    "phenotype_index",
    "phenotype_name",
]

OUTPUT_COLUMNS = [
    *KEY_COLUMNS,
    "secure_burden_p",
    "r_burden_p",
    "burden_abs_error",
    "secure_skat_wh_p",
    "r_skat_liu_p",
    "skat_liu_abs_error",
    "r_skat_davies_p",
    "r_skat_davies_converged",
    "skat_davies_abs_error",
]

R2_COMPARISONS = [
    (
        "Burden",
        "secure_burden_p",
        "r_burden_p",
        None,
    ),
    (
        "SKAT WH vs Liu",
        "secure_skat_wh_p",
        "r_skat_liu_p",
        None,
    ),
    (
        "SKAT WH vs Davies+fallback",
        "secure_skat_wh_p",
        "r_skat_davies_p",
        "r_skat_davies_converged",
    ),
]

def read_rows(
    path: Path,
    delimiter: str = ",",
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file, delimiter=delimiter))


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in KEY_COLUMNS)


def index_rows(
    rows: list[dict[str, str]],
    label: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}

    for row in rows:
        key = row_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {label} result key: {key}")
        indexed[key] = row

    return indexed


def has_test(row: dict[str, str]) -> bool:
    try:
        return math.isfinite(float(row["r_burden_p"]))
    except ValueError:
        return False


def absolute_error(left: str, right: str) -> str:
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return ""

    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return ""

    return format(abs(left_value - right_value), ".17g")


def r_squared_values(
    secure_values: list[float],
    reference_values: list[float],
) -> tuple[int, float | None]:
    if len(reference_values) < 2:
        return len(reference_values), None

    reference_mean = sum(reference_values) / len(reference_values)
    total_sum_squares = sum(
        (value - reference_mean) ** 2 for value in reference_values
    )
    if total_sum_squares == 0:
        return len(reference_values), None

    residual_sum_squares = sum(
        (secure - reference) ** 2
        for secure, reference in zip(secure_values, reference_values)
    )
    return len(reference_values), 1 - residual_sum_squares / total_sum_squares


def r_squared(
    rows: list[dict[str, str]],
    secure_column: str,
    reference_column: str,
) -> tuple[int, float | None]:
    secure_values = []
    reference_values = []

    for row in rows:
        try:
            secure_value = float(row[secure_column])
            reference_value = float(row[reference_column])
        except ValueError:
            continue

        if (
            math.isfinite(secure_value)
            and math.isfinite(reference_value)
            and secure_value >= 0
            and reference_value >= 0
        ):
            secure_values.append(
                -math.log10(max(secure_value, 1e-300))
            )
            reference_values.append(
                -math.log10(max(reference_value, 1e-300))
            )

    return r_squared_values(secure_values, reference_values)


def comparison_r_squared(
    rows: list[dict[str, str]],
    secure_column: str,
    reference_column: str,
    convergence_column: str | None,
) -> tuple[int, int, float | None, float | None]:
    total_count, all_score = r_squared(
        rows,
        secure_column,
        reference_column,
    )
    if convergence_column is None:
        return total_count, 0, all_score, all_score

    failed_count = sum(
        row[convergence_column] == "0"
        for row in rows
    )
    non_failed_rows = [
        row
        for row in rows
        if row[convergence_column] != "0"
    ]
    _, non_failed_score = r_squared(
        non_failed_rows,
        secure_column,
        reference_column,
    )
    return total_count, failed_count, all_score, non_failed_score


def print_r_squared_by_pheno_and_chr(
    rows: list[dict[str, str]],
) -> None:
    grouped_rows = {}

    for row in rows:
        key = (
            int(row["phenotype_index"]),
            row["phenotype_name"],
            int(row["chromosome"]),
        )
        grouped_rows.setdefault(key, []).append(row)

    print("\n=== R^2 on -log10(p) by phenotype and chromosome ===")
    print(
        f"  {'phenotype':<20} {'chr':>3} "
        f"{'comparison':<30} {'#gene (total(failed))':>21}"
        f"{'R^2 all':>12} {'R^2 non-failed':>16}"
    )

    for (_, phenotype_name, chromosome), group in sorted(
        grouped_rows.items()
    ):
        for (
            label,
            secure_column,
            reference_column,
            convergence_column,
        ) in R2_COMPARISONS:
            total, failed, all_score, non_failed_score = comparison_r_squared(
                group,
                secure_column,
                reference_column,
                convergence_column,
            )
            all_text = "NA" if all_score is None else f"{all_score:.6f}"
            non_failed_text = (
                "NA"
                if non_failed_score is None
                else f"{non_failed_score:.6f}"
            )
            gene_count = f"{total}({failed})"

            print(
                f"  {phenotype_name:<20} {chromosome:>3} "
                f"{label:<30} {gene_count:>21}"
                f"{all_text:>12} {non_failed_text:>16}"
            )


def compare_ancestry(
    run_dir: Path,
    ancestry: str,
) -> None:
    secure_rows = read_rows(
        run_dir / "secure" / ancestry / "all_secure_results.tsv",
        delimiter="\t",
    )
    reference_rows = read_rows(
        run_dir / "reference" / ancestry / "all_r_results.csv"
    )

    secure_by_key = index_rows(secure_rows, "secure")
    reference_by_key = index_rows(reference_rows, "reference")

    if set(secure_by_key) != set(reference_by_key):
        raise ValueError("secure and reference result keys do not match")

    genes = {row_key(row)[:3] for row in reference_rows}
    untested_genes = {row_key(row)[:3] for row in reference_rows if not has_test(row)}

    comparison_rows: list[dict[str, str]] = []

    for secure in secure_rows:
        reference = reference_by_key[row_key(secure)]

        comparison_rows.append(
            {
                **{
                    column: secure[column]
                    for column in KEY_COLUMNS
                },
                "secure_burden_p": secure["secure_burden_p"],
                "r_burden_p": reference["r_burden_p"],
                "burden_abs_error": absolute_error(
                    secure["secure_burden_p"],
                    reference["r_burden_p"],
                ),
                "secure_skat_wh_p": secure["secure_skat_wh_p"],
                "r_skat_liu_p": reference["r_skat_liu_p"],
                "skat_liu_abs_error": absolute_error(
                    secure["secure_skat_wh_p"],
                    reference["r_skat_liu_p"],
                ),
                "r_skat_davies_p":
                    reference["r_skat_davies_p"],
                "r_skat_davies_converged":
                    reference["r_skat_davies_converged"],
                "skat_davies_abs_error": absolute_error(
                    secure["secure_skat_wh_p"],
                    reference["r_skat_davies_p"],
                ),
            }
        )

    comparison_dir = run_dir / "comparison" / ancestry
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_path = comparison_dir / "all_comparison.csv"
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("Wrote secure/plain comparison to", output_path)
    print(f"\n=== {ancestry} ===")
    print(
        f"genes without a tested variant (no variable column in pooled inputs): "
        f"{len(untested_genes)} of {len(genes)}; their rows are left out of the R^2 and the plots"
    )
    print_r_squared_by_pheno_and_chr(comparison_rows)


def compare_secure_to_reference(config_path: Path) -> None:
    config = load_party_config(config_path)

    run_dir = Path(config["run_dir"])
    comparison_root = run_dir / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)

    success_path = comparison_root / "_SUCCESS"
    success_path.unlink(missing_ok=True)

    for ancestry in config["ancestries"]:
        compare_ancestry(run_dir, ancestry)

    success_path.touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/1kg-nparty",
        help="configuration directory",
    )
    args = parser.parse_args()

    compare_secure_to_reference(Path(args.config))


if __name__ == "__main__":
    main()
