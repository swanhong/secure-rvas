#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from config import load_party_config
from stage_timings import four_stage_timings, STAGE_SERIES

from compare_secure_to_reference import R2_COMPARISONS, comparison_r_squared


# Chromosome totals cover statistics only; alignment is shared across chromosomes.
CHROMOSOME_STAGES = (
    ("Weights", "compute_weights"),
    ("Stat", "packed_statistics"),
    ("Finalize", "finalize"),
    ("Release", "release"),
)

SESSION_SETUP_STAGES = (
    "network_init",
    "sample_count_exchange",
    "collective_setup",
)

COMMUNICATION_FIELDS = (
    "sent_bytes",
    "received_bytes",
)

SETUP_COMMUNICATION_STAGES = {
    "sample_count_exchange",
    "collective_setup",
    "null_model",
}

REQUIRED_METRIC_COLUMNS = {
    "stage",
    "chromosome",
    "duration_seconds",
    *COMMUNICATION_FIELDS,
}

def print_table(
    table: pd.DataFrame,
    left_columns: set[str],
) -> None:
    headers = [str(column) for column in table.columns]
    rows = [
        ["NA" if pd.isna(value) else str(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    widths = [
        max([len(header), *(len(row[index]) for row in rows)])
        for index, header in enumerate(headers)
    ]

    def print_row(row: list[str]) -> None:
        values = []
        for header, value, width in zip(headers, row, widths):
            if header in left_columns:
                values.append(f"{value:<{width}}")
            else:
                values.append(f"{value:>{width}}")
        print("  ".join(values))

    print_row(headers)
    print("  ".join("─" * width for width in widths))
    for row in rows:
        print_row(row)

def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h{int(minutes):02d}m{seconds:06.3f}s"

    if seconds >= 60:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m{seconds:06.3f}s"

    return f"{seconds:.3f}s"

def format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB")

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024

    raise AssertionError("unreachable")


def format_r_squared(score: float | None) -> str:
    if score is None:
        return "NA"
    return f"{score:.6f}"

def read_metrics(metric_paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in metric_paths:
        frame = pd.read_csv(path)

        missing_columns = REQUIRED_METRIC_COLUMNS - set(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing columns: {missing}")

        # Older runs named the alignment stages epoch, epoch_caps, epoch_lookup and epoch_embed.
        frame["stage"] = frame["stage"].str.replace(r"^epoch", "alignment", regex=True)

        if "ancestry" not in frame.columns:
            frame["ancestry"] = path.parent.name
        else:
            ancestry = frame["ancestry"].astype("string").str.strip()
            frame["ancestry"] = ancestry.mask(
                ancestry.isna() | ancestry.eq(""),
                path.parent.name,
            )

        frame["chromosome"] = (
            pd.to_numeric(frame["chromosome"])
            .fillna(0)
            .astype(int)
        )
        frame["duration_seconds"] = pd.to_numeric(
            frame["duration_seconds"]
        )

        for column in COMMUNICATION_FIELDS:
            frame[column] = (
                pd.to_numeric(frame[column])
                .fillna(0)
                .astype(int)
            )

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)

def make_alignment_time_table(metrics: pd.DataFrame) -> pd.DataFrame:
    # Only parent timers: coordinate substeps must not enter the total twice.
    phases = metrics["stage"].str.extract(r"^embedding_M(?P<M>[0-9]+)_(?P<phase>coordinates|finalize)$")
    selected = metrics.loc[phases["M"].notna()].copy()
    columns = ["Ancestry", "M", "Coordinates", "Finalize", "Total"]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["M"] = phases.loc[selected.index, "M"].astype(int)
    selected["phase"] = phases.loc[selected.index, "phase"]
    durations = selected.groupby(["ancestry", "M", "phase"])["duration_seconds"].sum().unstack(fill_value=0)
    durations = durations.reindex(columns=["coordinates", "finalize"], fill_value=0)
    rows = []
    for ancestry, group in durations.groupby(level="ancestry", sort=True):
        for (_, size), values in group.iterrows():
            rows.append([ancestry, str(size), *map(format_duration, values), format_duration(values.sum())])
        totals = group.sum()
        rows.append([ancestry, "TOTAL", *map(format_duration, totals), format_duration(totals.sum())])
    return pd.DataFrame(rows, columns=columns)


def make_chromosome_time_table(metrics: pd.DataFrame) -> pd.DataFrame:
    completed = metrics.loc[
        (metrics["chromosome"] != 0) & (metrics["stage"] == "chromosome_total"),
        ["ancestry", "chromosome"],
    ].drop_duplicates().sort_values(["ancestry", "chromosome"])
    if completed.empty:
        raise ValueError("no chromosome_total rows found")
    stages = [stage for _, stage in CHROMOSOME_STAGES]
    durations = (
        metrics.loc[metrics["chromosome"] != 0]
        .groupby(["ancestry", "chromosome", "stage"])["duration_seconds"].sum()
        .unstack(fill_value=0)
        .reindex(pd.MultiIndex.from_frame(completed))
        .reindex(columns=[*stages, "chromosome_total"], fill_value=0)
    )
    rows = []
    for ancestry, group in durations.groupby(level="ancestry", sort=True):
        for (_, chromosome), values in group.iterrows():
            rows.append([ancestry, str(chromosome), *map(format_duration, values)])
        rows.append([ancestry, "TOTAL", *map(format_duration, group.sum())])
    return pd.DataFrame(rows, columns=["Ancestry", "Chr", *(label for label, _ in CHROMOSOME_STAGES), "Total"])


def make_total_time_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numbers = []
    for ancestry, group in metrics.groupby("ancestry", sort=True):
        durations = group.groupby("stage")["duration_seconds"].sum()
        setup = sum(durations.get(stage, 0) for stage in SESSION_SETUP_STAGES)
        caps = durations.get("embedding_private_cap", 0) + durations.get("alignment_caps", 0)
        keygen = durations.get("embedding_keygen", 0)
        embedding = group.loc[
            group["stage"].str.match(r"^embedding_M[0-9]+_(coordinates|finalize)$"),
            "duration_seconds",
        ].sum()
        embedding += durations.get("alignment_lookup", 0) + durations.get("alignment_embed", 0)
        alignment = durations.get("alignment", 0)
        other = max(0, alignment - caps - keygen - embedding)
        null = durations.get("null_model", 0)
        statistics = durations.get("chromosome_total", 0)
        values = [setup, caps, keygen, embedding, other, null, statistics,
                  setup + alignment + null + statistics]
        numbers.append(values)
        rows.append([ancestry, *map(format_duration, values)])
    if len(numbers) > 1:
        rows.append(["ALL", *map(format_duration, map(sum, zip(*numbers)))])
    return pd.DataFrame(rows, columns=[
        "Ancestry", "Setup", "Caps", "Emb.Keygen", "Embedding", "Align.Other",
        "Null", "Statistics", "Total",
    ])


def make_four_stage_table(metrics: pd.DataFrame, lanes: int) -> pd.DataFrame:
    rows = []
    for ancestry, group in metrics.groupby("ancestry", sort=True):
        timings = four_stage_timings(group.to_dict("records"), lanes)
        rows.append([ancestry, *(f"{timings[key]:.6f}s" if timings[key] < 1 else format_duration(timings[key])
                                 for key, _, _ in STAGE_SERIES)])
    return pd.DataFrame(rows, columns=["Ancestry", *(label for _, label, _ in STAGE_SERIES)])


def make_comm_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Setup, Alignment (all chromosomes), one row per chromosome (its statistics), TOTAL."""
    columns = ["Ancestry", "Scope", "Sent", "Received", "Total I/O"]
    fields = list(COMMUNICATION_FIELDS)
    setup = (
        metrics.loc[
            (metrics["chromosome"] == 0)
            & metrics["stage"].isin(SETUP_COMMUNICATION_STAGES)
        ]
        .groupby("ancestry")[fields]
        .sum()
    )
    alignment = metrics.loc[metrics["stage"] == "alignment"].groupby("ancestry")[fields].sum()
    chromosomes = (
        metrics.loc[
            (metrics["chromosome"] != 0)
            & (metrics["stage"] == "chromosome_total")
        ]
        .groupby(["ancestry", "chromosome"], sort=True)[fields]
        .sum()
    )
    ancestries = sorted(
        set(setup.index)
        | set(alignment.index)
        | set(chromosomes.index.get_level_values("ancestry"))
    )
    if not ancestries:
        return pd.DataFrame(columns=columns)

    def comm_row(
        ancestry: str,
        scope: str,
        values: pd.Series,
    ) -> list[str]:
        sent = int(values["sent_bytes"])
        received = int(values["received_bytes"])
        return [
            ancestry,
            scope,
            format_bytes(sent),
            format_bytes(received),
            format_bytes(sent + received),
        ]

    zero = pd.Series({field: 0 for field in fields})
    rows = []
    grand = zero.copy()
    for ancestry in ancestries:
        total = zero.copy()
        for scope, table in (("Setup", setup), ("Alignment", alignment)):
            values = table.loc[ancestry] if ancestry in table.index else zero
            rows.append(comm_row(ancestry, scope, values))
            total = total + values
        if ancestry in chromosomes.index.get_level_values("ancestry"):
            for chromosome, values in chromosomes.loc[ancestry].sort_index().iterrows():
                rows.append(comm_row(ancestry, f"chr{chromosome}", values))
                total = total + values
        rows.append(comm_row(ancestry, "TOTAL", total))
        grand = grand + total

    if len(ancestries) > 1:
        rows.append(comm_row("ALL", "TOTAL", grand))

    return pd.DataFrame(rows, columns=columns)


def read_accuracy(
    comparison_specs: list[str],
) -> pd.DataFrame:
    frames = []

    required_columns = {
        "phenotype_index",
        "phenotype_name",
        "chromosome",
    }
    for _, secure, reference, convergence in R2_COMPARISONS:
        required_columns.update((secure, reference))
        if convergence is not None:
            required_columns.add(convergence)

    for spec in comparison_specs:
        ancestry, separator, path_text = spec.partition("=")
        ancestry = ancestry.strip()
        path_text = path_text.strip()
        if not separator or not ancestry or not path_text:
            raise ValueError(
                f"invalid comparison file {spec!r}; expected ANCESTRY=PATH"
            )

        path = Path(path_text)
        if not path.is_file():
            continue

        frame = pd.read_csv(path, dtype=str)
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing columns: {missing}")

        frame["ancestry"] = ancestry
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def make_accuracy_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ancestry",
        "Comparison",
        "#gene (total(failed))",
        "R^2 (all, R::SKAT)",
        "R^2 (non-failed)",
        "Worst R^2 (all)",
        "Worst phenotype",
        "Worst Chr",
        "Worst #gene (total(failed))",
    ]
    if comparisons.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for ancestry in comparisons["ancestry"].drop_duplicates():
        ancestry_rows = comparisons.loc[
            comparisons["ancestry"] == ancestry
        ]
        groups = [
            (
                (
                    int(group_key[0]),
                    str(group_key[1]),
                    int(group_key[2]),
                ),
                group,
            )
            for group_key, group in ancestry_rows.groupby(
                ["phenotype_index", "phenotype_name", "chromosome"],
                sort=True,
            )
        ]

        for label, secure, reference, convergence in R2_COMPARISONS:
            (
                pooled_count,
                pooled_failed,
                pooled_score,
                pooled_non_failed_score,
            ) = comparison_r_squared(
                ancestry_rows.to_dict("records"),
                secure,
                reference,
                convergence,
            )

            worst = None
            for group_key, group in groups:
                count, failed, score, _ = comparison_r_squared(
                    group.to_dict("records"),
                    secure,
                    reference,
                    convergence,
                )
                if score is None:
                    continue

                candidate = (score, *group_key, count, failed)
                if worst is None or candidate < worst:
                    worst = candidate

            if worst is None:
                worst_score = None
                worst_phenotype = "NA"
                worst_chromosome = "NA"
                worst_count = 0
                worst_failed = 0
            else:
                (
                    worst_score,
                    phenotype_index,
                    phenotype_name,
                    chromosome,
                    worst_count,
                    worst_failed,
                ) = worst
                worst_phenotype = f"{phenotype_index}:{phenotype_name}"
                worst_chromosome = str(chromosome)

            rows.append([
                ancestry,
                label,
                f"{pooled_count}({pooled_failed})",
                format_r_squared(pooled_score),
                format_r_squared(pooled_non_failed_score),
                format_r_squared(worst_score),
                worst_phenotype,
                worst_chromosome,
                f"{worst_count}({worst_failed})",
            ])

    return pd.DataFrame(rows, columns=columns)


def make_genomewide_r2_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    columns = ["Ancestry", "Phenotype", "Comparison", "#gene", "R^2"]
    rows = []
    if comparisons.empty:
        return pd.DataFrame(columns=columns)

    for (ancestry, index, name), group in comparisons.groupby(
        ["ancestry", "phenotype_index", "phenotype_name"],
        sort=False,
    ):
        records = group.to_dict("records")
        for label, secure, reference, convergence in R2_COMPARISONS:
            count, _, score, _ = comparison_r_squared(
                records, secure, reference, convergence
            )
            rows.append([
                ancestry,
                f"{index}:{name}",
                label,
                count,
                format_r_squared(score),
            ])

    return pd.DataFrame(rows, columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--party-id", type=int, default=1)
    parser.add_argument("--all-parties", action="store_true", help="summarize every data site P1..PN")
    parser.add_argument("--config-b", type=Path, help="B configuration directory for a combined A/B summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.config_b is not None and args.party_id != 1:
            raise ValueError("--config-b requires --party-id 1")
        if args.all_parties and (args.config_b is not None or args.party_id != 1):
            raise ValueError("--all-parties cannot be combined with --config-b or a different --party-id")
        sites = [(args.config, args.party_id)]
        if args.all_parties:
            count = len(load_party_config(args.config)["servers"]) - 1
            sites = [(args.config, party) for party in range(1, count + 1)]
        if args.config_b is not None:
            sites.append((args.config_b, 2))
        summaries = []
        for config_path, party_id in sites:
            config = load_party_config(config_path, party_id)
            configured_run_dir = str(config.get("run_dir", "")).strip()
            if not configured_run_dir:
                raise ValueError(f"run_dir is missing from {config_path}")
            run_dir = Path(configured_run_dir)
            metrics_dir = run_dir / "metrics"
            metric_paths = sorted(metrics_dir.rglob(f"metrics_party{party_id}.csv"))
            if not metric_paths:
                raise ValueError(f"no metrics_party{party_id}.csv files found under {metrics_dir}")
            metrics = read_metrics(metric_paths)
            try:
                four_stage = make_four_stage_table(metrics, config["mpc_num_threads"])
            except ValueError as error:
                four_stage = str(error)
            summaries.append((config_path, party_id, run_dir, make_alignment_time_table(metrics), make_chromosome_time_table(metrics), make_total_time_table(metrics), make_comm_table(metrics), four_stage))

        config = load_party_config(args.config, args.party_id)
        run_dir = summaries[0][2]
        ancestries = [
            str(ancestry).strip().upper()
            for ancestry in config.get("ancestries", [])
        ]
        comparison_specs = [
            (
                f"{ancestry}="
                f"{run_dir}/comparison/{ancestry}/all_comparison.csv"
            )
            for ancestry in ancestries
        ]

        accuracy = read_accuracy(comparison_specs)
        accuracy_table = make_accuracy_table(accuracy)
        genomewide_r2_table = make_genomewide_r2_table(accuracy)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(str(error)) from error

    print("Secure RVAS metrics summary")
    for config_path, party_id, run_dir, alignment_table, chromosome_table, total_table, comm_table, four_stage in summaries:
        print(f"\nConfig: {config_path}")
        print(f"Run directory: {run_dir}")
        print(f"\nAlignment by M (party{party_id}; excludes caps and keygen)")
        if alignment_table.empty:
            print("Unavailable: no per-M embedding timers in these metrics.")
        else:
            print_table(alignment_table, {"Ancestry"})
        print(f"\nChromosome statistics (party{party_id}; excludes alignment)")
        print_table(chromosome_table, {"Ancestry"})
        print(f"\nTotal timing summary (party{party_id})")
        print_table(total_table, {"Ancestry"})
        print("Total sums measured setup, alignment, null model and chromosome statistics; excludes input loading, result writing and wrapper overhead.")
        print(f"\nFour-stage elapsed timing (party{party_id}; excludes setup and input loading)")
        if isinstance(four_stage, str):
            print(f"Unavailable: {four_stage}")
        else:
            print_table(four_stage, {"Ancestry"})
            print("Stage 3-2 includes communication waits and orchestration; these four scopes do not sum to end-to-end wall time.")
        print(f"\nCommunication summary (party{party_id})")
        print_table(comm_table, {"Ancestry", "Scope"})

    print("\nR^2 summary on -log10(p)")
    if accuracy_table.empty:
        print("Unavailable: no comparison CSV files were found.")
    else:
        print_table(
            accuracy_table,
            {"Ancestry", "Comparison", "Worst phenotype"},
        )

    print("\nAll-chromosome R^2 by phenotype on -log10(p)")
    if genomewide_r2_table.empty:
        print("Unavailable: no comparison CSV files were found.")
    else:
        print_table(
            genomewide_r2_table,
            {"Ancestry", "Phenotype", "Comparison"},
        )

if __name__ == "__main__":
    main()
