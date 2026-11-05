from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class GeneRef:
    gene_id: str
    gene_symbol: str
    chromosome: str
    order_index: int


@dataclass(frozen=True)
class VariantRef:
    key: str
    position: int
    filter_value: str


@dataclass(frozen=True)
class AnnotationRef:
    variant_key: str
    gene_id: str
    gene_symbol: str
    values: Mapping[str, str]


@dataclass(frozen=True)
class GeneVariants:
    gene: GeneRef
    variants: tuple[VariantRef, ...]


@dataclass(frozen=True)
class GenePlan:
    gene: GeneRef
    variant_roles: tuple[
        tuple[VariantRef, int],  # 0: shared by all sites; 1..N: exclusive owner.
        ...,
    ]

@dataclass(frozen=True)
class GeneBlock:
    """One gene: keys and genotype columns for each data site, in party order."""
    gene: GeneRef
    keys: tuple[tuple[str, ...], ...]
    genotypes: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PhenoCovRows:
    sample_ids: tuple[str, ...]
    covariates: tuple[tuple[float, ...], ...]
    phenotypes: tuple[tuple[float, ...], ...]

@dataclass(frozen=True)
class PrepInputs:
    pgen_prefix: Path
    psam_ids: tuple[str, ...]
    gene_panel: tuple[GeneRef, ...]
    variants: tuple[VariantRef, ...]
    annotations: tuple[AnnotationRef, ...]
    annotation_columns: tuple[str, ...]

@dataclass(frozen=True)
class SampleInputs:
    phenotypes: Mapping[str, Mapping[str, float]]
    covariates: Mapping[str, tuple[float, ...]]
    ancestries: Mapping[str, str]
