#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import monotonic

from config import load_site_configs, site_input_paths


REFERENCE_COLUMNS = [
    "gene_index",
    "gene_id",
    "phenotype_index",
    "burden_p",
    "skat_davies_p",
    "skat_davies_converged",
    "skat_liu_p",
]

ENGINES = {
    "python": (
        "Python",
        sys.executable,
        "python_skat/compute_reference.py",
    ),
}

OUTPUT_COLUMNS = [
    "chromosome",
    "gene_index",
    "gene_id",
    "phenotype_index",
    "phenotype_name",
    "r_burden_p",
    "r_skat_davies_p",
    "r_skat_davies_converged",
    "r_skat_liu_p",
]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_chromosome_reference(
    sites: list[dict],
    reference_root: Path,
    engine_name: str,
    launcher: str,
    script: Path,
    ancestry: str,
    chromosome: int,
    phenotype_names: list[str],
    environment: dict[str, str],
) -> list[dict[str, str]]:
    print(
        f"Running {engine_name} reference for {ancestry} "
        f"chromosome {chromosome}",
        flush=True,
    )

    inputs = [site_input_paths(site, ancestry, chromosome) for site in sites]
    command = [launcher, str(script), "--inputs-json", json.dumps(inputs)]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )

    reader = csv.DictReader(
        completed.stdout.splitlines(),
        delimiter="\t",
    )
    if reader.fieldnames != REFERENCE_COLUMNS:
        raise ValueError(
            f"unexpected reference output columns: {reader.fieldnames}"
        )

    chromosome_rows: list[dict[str, str]] = []
    for row in reader:
        phenotype_index = int(row["phenotype_index"])
        if not 0 <= phenotype_index < len(phenotype_names):
            raise ValueError(
                f"invalid phenotype index {phenotype_index}"
            )

        chromosome_rows.append(
            {
                "chromosome": str(chromosome),
                "gene_index": row["gene_index"],
                "gene_id": row["gene_id"],
                "phenotype_index": row["phenotype_index"],
                "phenotype_name": phenotype_names[phenotype_index],
                "r_burden_p": row["burden_p"],
                "r_skat_davies_p": row["skat_davies_p"],
                "r_skat_davies_converged":
                    row["skat_davies_converged"],
                "r_skat_liu_p": row["skat_liu_p"],
            }
        )

    reference_dir = reference_root / ancestry
    reference_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        reference_dir / f"chr{chromosome}.csv",
        chromosome_rows,
    )
    return chromosome_rows


def worker_settings(engine: str, task_count: int) -> tuple[int, int]:
    cpu_count = os.cpu_count() or 1
    default_workers = min(16, cpu_count)
    workers = min(
        int(os.environ.get("REFERENCE_WORKERS", default_workers)),
        task_count,
        cpu_count,
    )
    blas_threads = int(
        os.environ.get("REFERENCE_BLAS_THREADS", 1)
    )
    workers = min(workers, max(1, cpu_count // blas_threads))
    return workers, blas_threads


def run_reference(config_path: Path, engine: str = "python", config_b: Path | None = None) -> None:
    sites = load_site_configs(config_path, config_b)
    config = sites[0]
    run_dir_a = Path(config["run_dir"])

    reference_root = run_dir_a / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)

    success_path = reference_root / "_SUCCESS"
    success_path.unlink(missing_ok=True)

    engine_name, launcher, script_name = ENGINES[engine]
    script = Path(__file__).resolve().parent / script_name
    tasks = [
        (ancestry, chromosome)
        for ancestry in config["ancestries"]
        for chromosome in config["chromosomes"]
    ]

    worker_count, blas_threads = worker_settings(engine, len(tasks))
    environment = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = str(blas_threads)

    print(
        f"Running {len(tasks)} {engine_name} reference tasks with "
        f"{worker_count} workers and {blas_threads} BLAS threads per worker"
    )

    reference_started_at = monotonic()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_tasks = {
            executor.submit(
                run_chromosome_reference,
                sites,
                reference_root,
                engine_name,
                launcher,
                script,
                ancestry,
                chromosome,
                config["phenotype_columns"],
                environment,
            ): (ancestry, chromosome)
            for ancestry, chromosome in tasks
        }
        task_rows = {}
        for completed_count, future in enumerate(
            as_completed(future_tasks),
            start=1,
        ):
            task = future_tasks[future]
            task_rows[task] = future.result()
            remaining_count = len(tasks) - completed_count
            elapsed_minutes = (
                monotonic() - reference_started_at
            ) / 60
            print(
                f"Reference progress: {completed_count} done / "
                f"{remaining_count} remaining "
                f"({elapsed_minutes:.1f} min elapsed)",
                flush=True,
            )

    rows_by_ancestry = {
        ancestry: [] for ancestry in config["ancestries"]
    }
    for ancestry, chromosome in tasks:
        rows_by_ancestry[ancestry].extend(
            task_rows[(ancestry, chromosome)]
        )

    for ancestry, ancestry_rows in rows_by_ancestry.items():
        output_path = reference_root / ancestry / "all_r_results.csv"
        write_csv(output_path, ancestry_rows)
        print(f"Wrote {engine_name} reference results to", output_path)

    success_path.touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/1kg-nparty",
        help="configuration directory",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="python",
        help="reference engine (default: python)",
    )
    parser.add_argument("--config-b", help="B configuration directory; defaults to the same inputs as --config")
    args = parser.parse_args()

    run_reference(Path(args.config), args.engine, Path(args.config_b) if args.config_b else None)


if __name__ == "__main__":
    main()
