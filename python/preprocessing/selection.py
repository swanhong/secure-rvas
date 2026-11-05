
import random
from collections.abc import Collection, Sequence

from .model import GeneRef, GeneVariants, PrepInputs

def select_random_gene_groups(
        gene_groups: Sequence[GeneVariants],
        gene_count: int,
        seed: int,
) -> tuple[GeneVariants, ...]:
    if gene_count < 1:
        raise ValueError("random count must be positive")
    candidates = tuple(
        group for group in gene_groups if group.variants
    )

    if gene_count > len(candidates):
        raise ValueError(
            f"random count {gene_count} exceeds available genes "
            f"with variants {len(candidates)}"
        )

    selected_ids = {
        group.gene.gene_id
        for group in random.Random(seed).sample(
            candidates,
            gene_count,
        )
    }

    return tuple(
        group for group in candidates if group.gene.gene_id in selected_ids
    )

def select_file_genes(
    inputs: PrepInputs,
    chromosome: str,
    selected_genes: Collection[GeneRef],
) -> tuple[GeneRef, ...]:
    requested = tuple(
        gene
        for gene in selected_genes
        if gene.chromosome == chromosome
    )

    requested_ids = set()
    for gene in requested:
        if gene.gene_id in requested_ids:
            raise ValueError(
                f"duplicate selected gene {gene.gene_id} "
                f"on chromosome {chromosome}"
            )
        requested_ids.add(gene.gene_id)

    available = tuple(
        gene
        for gene in inputs.gene_panel
        if (
            gene.chromosome == chromosome
            and gene.gene_id in requested_ids
        )
    )

    available_ids = {
        gene.gene_id
        for gene in available
    }
    missing = requested_ids - available_ids
    if missing:
        raise ValueError(
            f"selected genes not found on chromosome {chromosome}: "
            f"{', '.join(sorted(missing))}"
        )

    return available