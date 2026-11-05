#!/usr/bin/env python3

import argparse
import csv
import math
import tomllib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_secure_to_reference import read_rows, r_squared
from config import load_party_config


SCATTER_COMPARISONS = [
    ("burden", "Burden", "secure_burden_p", "r_burden_p"),
    (
        "skat_liu",
        "SKAT WH vs Reference Liu",
        "secure_skat_wh_p",
        "r_skat_liu_p",
    ),
    (
        "skat_davies",
        "SKAT WH vs Reference Davies (Liu fallback)",
        "secure_skat_wh_p",
        "r_skat_davies_p",
    ),
]
MANHATTAN_COMPARISONS = [
    (
        "burden",
        "Burden",
        "secure_burden_p",
        "Secure Burden",
        "r_burden_p",
        "Plain Burden",
    ),
    (
        "skat_liu",
        "SKAT",
        "secure_skat_wh_p",
        "Secure SKAT WH",
        "r_skat_liu_p",
        "Plain SKAT Liu",
    ),
    (
        "skat_davies",
        "SKAT",
        "secure_skat_wh_p",
        "Secure SKAT WH",
        "r_skat_davies_p",
        "Plain SKAT Davies+fallback",
    ),
]


def negative_log10(p_value: float) -> float:
    return -math.log10(max(p_value, 1e-300))


def valid_p_value(row: dict[str, str], column: str) -> float | None:
    try:
        p_value = float(row[column])
    except ValueError:
        return None
    return p_value if math.isfinite(p_value) and p_value >= 0 else None


def read_gene_symbols(config_path: Path, chromosomes: list[int]) -> dict[str, str]:
    with (config_path / "configPrepare.toml").open("rb") as config_file:
        panel_template = tomllib.load(config_file)["gene_panel"]

    gene_symbols = {}
    for chromosome in chromosomes:
        panel_path = Path(panel_template.format(chromosome=chromosome))
        with panel_path.open(newline="") as panel_file:
            for row in csv.DictReader(panel_file, delimiter="\t"):
                gene_id = row["gene_id"]
                gene_symbols[gene_id] = row["gene_symbol"].strip() or gene_id
    return gene_symbols


def select_label_gene_ids(
    rows: list[tuple[int, int, dict[str, str]]],
    secure_column: str,
    reference_column: str,
    threshold: float,
) -> set[str]:
    ranked_reference = []
    selected = set()
    for chromosome, gene_index, row in rows:
        secure_p = valid_p_value(row, secure_column)
        reference_p = valid_p_value(row, reference_column)
        if reference_p is not None:
            ranked_reference.append(
                (reference_p, chromosome, gene_index, row["gene_id"])
            )
        if any(
            p_value is not None and p_value <= threshold
            for p_value in (secure_p, reference_p)
        ):
            selected.add(row["gene_id"])

    selected.update(item[3] for item in sorted(ranked_reference)[:10])
    return selected


def place_gene_labels(figure, axis, points, obstacles=()) -> None:
    initial_offset = 3.0
    annotations = [
        (
            y_value,
            axis.annotate(
                label,
                xy=(x_value, y_value),
                xytext=(0, initial_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="black",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "linewidth": 0,
                    "pad": 0.15,
                },
                annotation_clip=False,
                zorder=4,
            ),
        )
        for x_value, y_value, label in points
    ]

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    placed_boxes = [box.expanded(1.02, 1.04) for box in obstacles]
    pixels_per_point = figure.dpi / 72.0

    for _, annotation in sorted(annotations, key=lambda item: item[0]):
        offset = initial_offset
        while True:
            annotation.set_position((0, offset))
            box = annotation.get_window_extent(renderer).expanded(1.02, 1.08)
            overlaps = [other for other in placed_boxes if box.overlaps(other)]
            if not overlaps:
                break
            offset += (
                max(other.y1 - box.y0 for other in overlaps) + 2.0
            ) / pixels_per_point

        placed_boxes.append(box)
        if offset == initial_offset:
            continue
        axis.annotate(
            "",
            xy=annotation.xy,
            xytext=(0, offset),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "-",
                "color": "#a6a6a6",
                "linewidth": 0.6,
            },
            annotation_clip=False,
            zorder=2,
        )


def negative_log10_p_value_pairs(
    rows: list[dict[str, str]],
    secure_column: str,
    reference_column: str,
) -> tuple[list[float], list[float]]:
    pairs = []
    for row in rows:
        secure_p = valid_p_value(row, secure_column)
        reference_p = valid_p_value(row, reference_column)
        if secure_p is not None and reference_p is not None:
            pairs.append((negative_log10(secure_p), negative_log10(reference_p)))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def write_scatter_plot(
    rows: list[dict[str, str]],
    secure_column: str,
    reference_column: str,
    title: str,
    output_path: Path,
) -> None:
    secure_values, reference_values = negative_log10_p_value_pairs(
        rows, secure_column, reference_column
    )
    count, score = r_squared(rows, secure_column, reference_column)
    score_text = "NA" if score is None else f"{score:.6f}"

    values = [0.0, *secure_values, *reference_values]
    lower, upper = min(values), max(values)
    padding = max(0.02 * (upper - lower), 0.02)
    limits = [lower - padding, upper + padding]

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(reference_values, secure_values, color="#2f5597", s=24)
    axis.plot(limits, limits, color="#c00000", linestyle="--", linewidth=1)
    axis.set(xlim=limits, ylim=limits)
    axis.set_xlabel("Reference -log10(p)")
    axis.set_ylabel("Secure -log10(p)")
    axis.set_title(f"{title}\nn={count}, $R^2$ (-log10 p)={score_text}")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_manhattan_plot(
    rows: list[dict[str, str]],
    secure_column: str,
    secure_label: str,
    reference_column: str,
    reference_label: str,
    title: str,
    output_path: Path,
    gene_symbols: dict[str, str],
) -> None:
    positioned_rows = sorted(
        (
            (int(row["chromosome"]), int(row["gene_index"]), row)
            for row in rows
        ),
        key=lambda item: item[:2],
    )
    count, score = r_squared(rows, secure_column, reference_column)
    score_text = "NA" if score is None else f"{score:.6f}"
    chromosome_indices = defaultdict(list)
    for index, (chromosome, _, _) in enumerate(positioned_rows):
        chromosome_indices[chromosome].append(index)
    chromosomes = list(chromosome_indices)
    colors = {
        chromosome: "#2f5597" if index % 2 == 0 else "#70ad47"
        for index, chromosome in enumerate(chromosomes)
    }

    threshold = 0.05 / len(positioned_rows)
    bonferroni = negative_log10(threshold)
    selected_ids = select_label_gene_ids(
        positioned_rows, secure_column, reference_column, threshold
    )
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
    panel_label_points = []
    plot_peak = bonferroni

    for axis, (column, panel_label) in zip(
        axes,
        ((secure_column, secure_label), (reference_column, reference_label)),
    ):
        points = []
        for index, (chromosome, _, row) in enumerate(positioned_rows):
            p_value = valid_p_value(row, column)
            if p_value is not None:
                points.append(
                    (
                        index,
                        negative_log10(p_value),
                        colors[chromosome],
                        row["gene_id"],
                    )
                )
        selected = [point for point in points if point[3] in selected_ids]
        if points:
            plot_peak = max(plot_peak, max(point[1] for point in points))

        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            c=[point[2] for point in points],
            s=22,
        )
        axis.scatter(
            [point[0] for point in selected],
            [point[1] for point in selected],
            c=[point[2] for point in selected],
            s=22,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        axis.axhline(
            bonferroni,
            color="#c00000",
            linestyle="--",
            linewidth=1,
            label=f"Bonferroni 0.05/{len(positioned_rows)}",
        )
        axis.set_ylabel("-log10(p)")
        axis.set_title(panel_label)
        axis.grid(axis="y", alpha=0.2)
        for chromosome in chromosomes[:-1]:
            axis.axvline(
                max(chromosome_indices[chromosome]) + 0.5,
                color="#bfbfbf",
                linewidth=0.7,
            )
        panel_label_points.append(
            [
                (x_value, y_value, gene_symbols.get(gene_id, gene_id))
                for x_value, y_value, _, gene_id in selected
            ]
        )

    legend = axes[0].legend(loc="upper right")
    axes[-1].set_xticks(
        [sum(indices) / len(indices) for indices in chromosome_indices.values()],
        [f"chr{chromosome}" for chromosome in chromosomes],
        fontsize=9,
    )
    axes[-1].set_xlabel("Gene order")
    figure.suptitle(
        f"{title}\nn={count}, $R^2$ (-log10 p)={score_text}",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    lower = axes[0].get_ylim()[0]
    axes[0].set_ylim(lower, plot_peak + 0.15 * (plot_peak - lower))
    figure.canvas.draw()
    legend_box = legend.get_window_extent(figure.canvas.get_renderer())
    for index, (axis, points) in enumerate(zip(axes, panel_label_points)):
        obstacles = (legend_box,) if index == 0 else ()
        place_gene_labels(figure, axis, points, obstacles)

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_ancestry(run_dir: Path, ancestry: str, gene_symbols: dict[str, str]) -> None:
    comparison_dir = run_dir / "comparison" / ancestry
    plots_dir = comparison_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    success_path = plots_dir / "_SUCCESS"
    success_path.unlink(missing_ok=True)

    chromosome_groups = defaultdict(list)
    phenotype_groups = defaultdict(list)
    for row in read_rows(comparison_dir / "all_comparison.csv"):
        phenotype = (int(row["phenotype_index"]), row["phenotype_name"])
        chromosome_groups[(*phenotype, int(row["chromosome"]))].append(row)
        phenotype_groups[phenotype].append(row)

    written = 0
    for (phenotype_index, phenotype_name, chromosome), rows in sorted(
        chromosome_groups.items()
    ):
        for tag, label, secure_column, reference_column in SCATTER_COMPARISONS:
            write_scatter_plot(
                rows,
                secure_column,
                reference_column,
                f"{label}: {phenotype_name}, {ancestry}, chromosome {chromosome}",
                plots_dir
                / f"scatter_{tag}_pheno{phenotype_index}_chr{chromosome}.png",
            )
            written += 1

    for (phenotype_index, phenotype_name), rows in sorted(
        phenotype_groups.items()
    ):
        for comparison in MANHATTAN_COMPARISONS:
            tag, label, secure_column, secure_label, reference_column, reference_label = (
                comparison
            )
            write_manhattan_plot(
                rows,
                secure_column,
                secure_label,
                reference_column,
                reference_label,
                f"{label}: {phenotype_name}, {ancestry}",
                plots_dir / f"manhattan_{tag}_pheno{phenotype_index}.png",
                gene_symbols,
            )
            written += 1

    success_path.touch()
    print(f"Wrote {written} plots to {plots_dir}")


def plot_secure_vs_reference(config_path: Path) -> None:
    config = load_party_config(config_path)
    gene_symbols = read_gene_symbols(config_path, config["chromosomes"])
    run_dir = Path(config["run_dir"])
    success_path = run_dir / "comparison" / "_PLOTS_SUCCESS"
    success_path.unlink(missing_ok=True)
    for ancestry in config["ancestries"]:
        plot_ancestry(run_dir, ancestry, gene_symbols)
    success_path.touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/1kg-nparty", help="configuration directory")
    args = parser.parse_args()
    plot_secure_vs_reference(Path(args.config))


if __name__ == "__main__":
    main()
