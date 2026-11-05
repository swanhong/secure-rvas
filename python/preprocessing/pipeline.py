import math
import random
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence, Callable
from pathlib import Path
from typing import Literal
import numpy as np

from .model import (
    AnnotationRef,
    GeneBlock,
    GeneRef,
    GeneVariants,
    PhenoCovRows,
    VariantRef,
    GenePlan,
)

from .output import write_outputs
from .plink import pgen_extract

GenotypeExtractor = Callable[
    [Path, tuple[str, ...], tuple[str, ...]],
    tuple[np.ndarray, tuple[str, ...]],
]


def require_columns(
        table: Mapping[str, Mapping[str, float]],
        columns: Sequence[str],
        label: str,
) -> None:
    row = next(iter(table.values()), None)
    if row is None:
        raise ValueError(f"{label} table is empty")

    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(
            f"{label} columns not found: {', '.join(missing)}"
        )

def extract_genotypes(
    pgen_prefix: Path,
    rows: Sequence[PhenoCovRows],
    plans: Sequence[GenePlan],
    extractor: GenotypeExtractor,
) -> tuple[tuple[GenePlan, ...], tuple[np.ndarray, ...], tuple[dict[str, int], ...]]:
    """Extract each site's columns; retain shared variants only if all sites emitted them."""
    genotypes, columns = [], []
    for party, local in enumerate(rows, 1):
        keys = tuple(dict.fromkeys(
            variant.key for plan in plans for variant, owner in plan.variant_roles
            if owner in (0, party)
        ))
        matrix, emitted = extractor(pgen_prefix, local.sample_ids, keys)
        genotypes.append(matrix)
        columns.append({key: col for col, key in enumerate(emitted)})
    filtered = []
    for plan in plans:
        retained = []
        for variant, owner in plan.variant_roles:
            holders = columns if owner == 0 else (columns[owner - 1],)
            if all(variant.key in column for column in holders):
                retained.append((variant, owner))
        filtered.append(GenePlan(plan.gene, tuple(retained)))
    return tuple(filtered), tuple(genotypes), tuple(columns)


def build_blocks(
    plans: Sequence[GenePlan],
    genotypes: Sequence[np.ndarray],
    key_columns: Sequence[Mapping[str, int]],
    keep_zero_variants: bool = False,
) -> tuple[GeneBlock, ...]:
    blocks = []
    for plan in plans:
        site_keys, site_genotypes = [], []
        for party, (matrix, columns) in enumerate(zip(genotypes, key_columns), 1):
            keys = tuple(variant.key for variant, owner in plan.variant_roles if owner in (0, party))
            block = matrix[:, [columns[key] for key in keys]]
            if not keep_zero_variants:
                carried = np.any(block != 0, axis=0)
                keys = tuple(key for key, keep in zip(keys, carried) if keep)
                block = block[:, carried]
            site_keys.append(keys)
            site_genotypes.append(block)
        blocks.append(GeneBlock(plan.gene, tuple(site_keys), tuple(site_genotypes)))
    return tuple(blocks)


def assign_roles(
    gene_variants: Sequence[GeneVariants],
    role_seed: int = 42,
    shared_rate: float = 0.6,
    num_sites: int = 2,
) -> tuple[GenePlan, ...]:
    if num_sites < 2 or not 0 <= shared_rate <= 1:
        raise ValueError("need num_sites >= 2 and shared_rate in [0, 1]")
    rng = random.Random(role_seed)
    plans = []
    for group in gene_variants:
        shuffled = list(group.variants)
        rng.shuffle(shuffled)
        shared_count = round(shared_rate * len(shuffled))
        owners = list(range(1, num_sites + 1))
        rng.shuffle(owners)  # Distribute rounding remainders without always favoring P1.
        role = {variant: 0 if j < shared_count else owners[(j - shared_count) % num_sites]
                for j, variant in enumerate(shuffled)}
        plans.append(GenePlan(group.gene, tuple((variant, role[variant]) for variant in group.variants)))
    return tuple(plans)


def select_gene_variants(
        gene_panel: Sequence[GeneRef],
        variants: Sequence[VariantRef],
        annotations: Sequence[AnnotationRef],
        annotation_columns: Collection[str],
        chromosome: str,
        gene_selection: Collection[str] | Literal["all"],
        mask: Mapping[str, str | Collection[str]],
        max_maf: float | None
) -> tuple[GeneVariants, ...]:
    """Select variants for genes in a gene panel.

    Args:
        gene_panel: Genes to consider.
        variants: Variants to consider.
        annotations: Annotations for the variants.
        annotation_columns: Annotation columns to use for filtering.
        chromosome: Chromosome to filter on.
        gene_selection: Gene IDs to select. If "all", all genes are selected.
        mask: chosen annotation values to filter on.
            input example: {"LoF": "HC", "consequence": {"missense_variant", "stop_gained"}}
        max_maf: Maximum minor allele frequency to filter variants. If None, no filtering is applied.

    Returns:
        A tuple of GeneVariants objects
    """
    for column in mask:
        if column not in annotation_columns:
            raise ValueError(f"mask column not found: {column}")

    if max_maf is not None and "MAF" not in annotation_columns:
        raise ValueError("MAF not found")

    accepted_values_by_column = {
        column: (
            {accepted_values}
            if isinstance(accepted_values, str)
            else set(accepted_values)
        ) for column, accepted_values in mask.items()
    }

    # A bare str is a Collection[str], so `in` below would substring-match.
    if isinstance(gene_selection, str) and gene_selection != "all":
        raise TypeError(
            "gene_selection must be 'all' or a collection of gene ids"
        )

    selected_genes = tuple(
        gene for gene in gene_panel
        if gene.chromosome == chromosome and (
            gene_selection == "all" or gene.gene_id in gene_selection
        )
    )
    selected_gene_ids = {
        gene.gene_id for gene in selected_genes
    }

    annotations_by_variant = {}
    seen_occurrences = set()
    for annotation in annotations:
        if annotation.gene_id not in selected_gene_ids:
            continue

        occurrence = (annotation.gene_id, annotation.variant_key)
        if occurrence in seen_occurrences:
            raise ValueError(
                f"duplicate annotation row for gene {annotation.gene_id} "
                f"and variant {annotation.variant_key}"
            )
        seen_occurrences.add(occurrence)

        annotations_by_variant.setdefault(
            annotation.variant_key,
            [],
        ).append(annotation)

    variants_by_gene = {
        gene.gene_id:[] for gene in selected_genes
    }

    for variant in variants:
        if variant.filter_value not in {"PASS", "."}:
            continue

        for annotation in annotations_by_variant.get(variant.key, ()):
            if any(
                annotation.values[column] not in accepted_values
                for column, accepted_values in accepted_values_by_column.items()
            ):
                continue
            if max_maf is not None and float(annotation.values["MAF"]) > max_maf:
                continue
            variants_by_gene[annotation.gene_id].append(variant)

    return tuple(
        GeneVariants(
            gene=gene,
            variants=tuple(variants_by_gene[gene.gene_id]),
        )
        for gene in selected_genes
    )

def select_rows(
    psam_ids: Sequence[str],
    phenotypes: Mapping[str, Mapping[str, float]],
    covariates: Mapping[str, tuple[float, ...]],
    ancestries: Mapping[str, str],
    ancestry: str,
    phenotype_columns: Sequence[str],
    samples_per_cohort: int | Literal["all"],
    sample_seed: int=42,
    party: int=0,
    num_sites: int=2,
) -> tuple[PhenoCovRows, ...]:
    if num_sites < 2 or not 0 <= party <= num_sites:
        raise ValueError("need at least two data sites and party in 0..num_sites")
    print("Selecting rows for cohorts")
    print("  ancestry:", ancestry)
    print("  phenotype columns:", ", ".join(phenotype_columns))
    print("  samples per cohort:", samples_per_cohort)
    print("  sample seed:", sample_seed)
    # filter samples
    #   - have both have both phenotypes and covariates,
    #   - all values are finite (not nan, inf, -inf, ...)
    require_columns(phenotypes, phenotype_columns, "phenotype")

    eligible = []
    for sample_id in psam_ids:
        if (
            sample_id not in phenotypes
            or sample_id not in covariates
            or ancestries.get(sample_id) != ancestry
        ):
            continue

        phenotype_values = tuple(
            phenotypes[sample_id][column]
            for column in phenotype_columns
        )
        covariate_values = covariates[sample_id]

        if not all(
            math.isfinite(value)
            for value in phenotype_values
        ):
            continue

        if not all(
            math.isfinite(value)
            for value in covariate_values
        ):
            continue

        eligible.append(
            (
                sample_id,
                covariate_values,
                phenotype_values,
            )
        )

    random.Random(sample_seed).shuffle(eligible)

    if party:
        count = len(eligible) if samples_per_cohort == "all" else samples_per_cohort
        if count <= 0 or len(eligible) < count:
            raise ValueError(f"need {count} eligible samples, found {len(eligible)}")
        counts = [count if site == party else 0 for site in range(1, num_sites + 1)]
    elif samples_per_cohort == "all":
        if len(eligible) < num_sites:
            raise ValueError(
                f"need {num_sites} eligible samples, found {len(eligible)}"
            )
        count, extra = divmod(len(eligible), num_sites)
        counts = [count + (site < extra) for site in range(num_sites)]
    else:
        if samples_per_cohort <= 0:
            raise ValueError(
                "samples_per_cohort must be positive or 'all'"
            )

        required = num_sites * samples_per_cohort
        if len(eligible) < required:
            raise ValueError(
                f"need {required} eligible samples, "
                f"found {len(eligible)}"
            )

        counts = [samples_per_cohort] * num_sites

    rows, start = [], 0
    for count in counts:
        selected = eligible[start:start + count]
        rows.append(PhenoCovRows(
            sample_ids=tuple(row[0] for row in selected),
            covariates=tuple(row[1] for row in selected),
            phenotypes=tuple(row[2] for row in selected),
        ))
        start += count
    return tuple(rows)

def prepare_blocks(
    pgen_prefix: Path,
    gene_variants: Sequence[GeneVariants],
    rows: Sequence[PhenoCovRows],
    role_seed: int,
    out_dir: Path,
    shared_rate: float = 0.6,
    keep_zero_variants: bool = False,
    extractor: GenotypeExtractor = pgen_extract,
) -> Path:
    plans = assign_roles(gene_variants, role_seed, shared_rate, len(rows))
    print("Reading PGEN:", out_dir, flush=True)
    extracted, genotypes, columns = extract_genotypes(pgen_prefix, rows, plans, extractor)
    blocks = build_blocks(extracted, genotypes, columns, keep_zero_variants)
    write_outputs(out_dir, blocks, rows)
    statistics = []
    for plan, retained, block in zip(plans, extracted, blocks):
        occurrences = Counter(key for local in block.keys for key in local)
        statistics.append(dict(
            gene=plan.gene.gene_id,
            assigned_union=len(plan.variant_roles),
            assigned_shared=sum(owner == 0 for _, owner in plan.variant_roles),
            unextractable=len(plan.variant_roles) - len(retained.variant_roles),
            union=len(occurrences),
            shared_all=sum(count == len(rows) for count in occurrences.values()),
            partial_shared=sum(1 < count < len(rows) for count in occurrences.values()),
            local_counts=[len(local) for local in block.keys],
            exclusive=[sum(occurrences[key] == 1 for key in local) for local in block.keys],
        ))
    (out_dir / "split.json").write_text(json.dumps(dict(
        num_sites=len(rows), shared_rate=shared_rate, role_seed=role_seed,
        keep_zero_variants=keep_zero_variants, genes=statistics,
    ), indent=2) + "\n")
    return out_dir
