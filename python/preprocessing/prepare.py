from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TextIO
import fcntl
import hashlib
import json
from tempfile import TemporaryDirectory
import sys

from .input import (
    load_inputs,
    load_sample_inputs,
    read_gene_panel,
)
from .model import (
    GeneRef,
    PrepInputs,
    GeneVariants,
)
from .pipeline import (
    GenotypeExtractor,
    prepare_blocks,
    select_gene_variants,
    select_rows,
)
from .plink import pgen_extract
from .output import write_selected_genes
from .site import prepare_site_blocks
from .selection import (
    select_file_genes,
    select_random_gene_groups,
)

@dataclass(frozen=True)
class GeneSelectionRequest:
    mode: Literal["random", "file", "all"]
    per_chromosome: int
    seed: int
    path: Path | None


@dataclass(frozen=True)
class PrepareRequest:
    run_dir: Path
    chromosomes: tuple[int, ...]

    genotype: str
    gene_panel: str
    annotation: str
    phenotype: Path
    covariate: Path
    ancestry: Path
    plink2_bin: str

    phenotype_id_column: str
    covariate_id_column: str
    covariate_column: str
    covariate_columns: tuple[str, ...]
    is_cov_single_column: bool
    ancestry_id_column: str
    ancestry_column: str
    phenotype_columns: tuple[str, ...]
    ancestries: tuple[str, ...]
    num_cov: int
    mask: Mapping[str, str | Collection[str]]
    max_maf: float | None

    samples_per_cohort: int
    sample_seed: int
    role_seed: int
    shared_rate: float

    gene_selection: GeneSelectionRequest
    prepared_cache_dir: Path | None = None
    party: int = 0
    num_sites: int = 2
    keep_zero_variants: bool = False


def chromosome_path(
    template: str,
    chromosome: int,
) -> Path:
    return Path(
        template.replace(
            "{chromosome}",
            str(chromosome),
        )
    )


def select_gene_groups(
    request: GeneSelectionRequest,
    inputs: PrepInputs,
    chromosome: int,
    mask: Mapping[str, str | Collection[str]],
    max_maf: float | None,
    file_genes: Collection[GeneRef],
) -> tuple[GeneVariants, ...]:
    chromosome_name = str(chromosome)

    if request.mode == "random":
        all_groups = select_gene_variants(
            gene_panel=inputs.gene_panel,
            variants=inputs.variants,
            annotations=inputs.annotations,
            annotation_columns=inputs.annotation_columns,
            chromosome=chromosome_name,
            gene_selection="all",
            mask=mask,
            max_maf=max_maf,
        )
        return select_random_gene_groups(
            gene_groups=all_groups,
            gene_count=request.per_chromosome,
            seed=request.seed,
        )

    if request.mode == "file":
        genes = select_file_genes(
            inputs=inputs,
            chromosome=chromosome_name,
            selected_genes=file_genes,
        )
        gene_selection = tuple(gene.gene_id for gene in genes)
    elif request.mode == "all":
        gene_selection = "all"
    else:
        raise ValueError(
            f"unsupported gene selection mode: {request.mode}"
        )
    return select_gene_variants(
        gene_panel=inputs.gene_panel,
        variants=inputs.variants,
        annotations=inputs.annotations,
        annotation_columns=inputs.annotation_columns,
        chromosome=chromosome_name,
        gene_selection=gene_selection,
        mask=mask,
        max_maf=max_maf,
    )


def prepared_cache_config(request: PrepareRequest, chromosome: int, ancestry: str) -> dict:
    """Identify prepared inputs without reading the genotype file contents."""
    def source(path: Path) -> dict:
        path = path.resolve()
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    config = asdict(request)
    for name in ("run_dir", "prepared_cache_dir", "chromosomes", "ancestries", "plink2_bin"):
        del config[name]
    prefix = chromosome_path(request.genotype, chromosome)
    config["genotype"] = {
        extension: source(Path(f"{prefix}.{extension}"))
        for extension in ("pgen", "pvar", "psam")
    }
    for name in ("annotation", "gene_panel"):
        config[name] = source(chromosome_path(getattr(request, name), chromosome))
    for name in ("phenotype", "ancestry"):
        config[name] = source(getattr(request, name))
    config["covariate"] = source(Path(str(request.covariate).replace("{anc}", ancestry.lower())))
    selection = config["gene_selection"]
    selection["path"] = source(selection["path"]) if selection["mode"] == "file" else None
    if not request.party:
        del config["party"]
    config["mask"] = {
        column: sorted({values} if isinstance(values, str) else set(values))
        for column, values in request.mask.items()
    }
    config.update(format_version=4, chromosome=chromosome, ancestry_group=ancestry)
    return config


def prepare_cached_blocks(
    request: PrepareRequest, chromosome: int, ancestry: str, build: Callable[..., Path],
) -> Path:
    cache_root = request.prepared_cache_dir.resolve()
    if cache_root.is_relative_to(request.run_dir.resolve()):
        raise ValueError("prepared_cache_dir must be outside run_dir")
    output = request.run_dir / "prepared" / ancestry / f"chr{chromosome}"
    if output.exists() and not output.is_symlink():
        raise ValueError(f"{output} is not a link; use a new run_dir or prepare --clear")
    config = json.dumps(prepared_cache_config(request, chromosome, ancestry), sort_keys=True, indent=2) + "\n"
    full_key = hashlib.sha256(config.encode()).hexdigest()
    key = full_key[:10]
    parent = cache_root / ancestry / f"chr{chromosome}"
    parent.mkdir(parents=True, exist_ok=True)
    cached = parent / key
    if not cached.exists() and (parent / full_key).exists():
        cached = parent / full_key
    if cached.exists():
        if (cached / "config.json").read_text() != config:
            raise ValueError(f"prepared cache config mismatch: {cached}")
        print(f"Using existing prepared cache: {cached}", flush=True)
        print(f"    ancestry: {ancestry}, chromosome: {chromosome}, MAF: {request.max_maf}, shared_rate: {request.shared_rate}", flush=True)
        print(f"    key: {key}", flush=True)
    else:
        print("Creating prepared cache:", cached, flush=True)
        with TemporaryDirectory(prefix=f".{key}.", dir=parent) as temporary:
            directory = Path(temporary)
            build(out_dir=directory)
            (directory / "config.json").write_text(config)
            directory.rename(cached)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        output.unlink()
    output.symlink_to(cached, target_is_directory=True)
    return output


def prepare_chromosomes(
    request: PrepareRequest,
    extractor: GenotypeExtractor | None = None,
) -> tuple[GeneRef, ...]:
    if not request.chromosomes:
        raise ValueError("at least one chromosome is required")

    if request.num_sites < 2 or not 0 <= request.party <= request.num_sites:
        raise ValueError("need num_sites >= 2 and party in 0..num_sites")
    if not 0 <= request.shared_rate <= 1:
        raise ValueError("shared_rate must be in [0, 1]")
    if request.party and request.gene_selection.mode == "random":
        raise ValueError("site preparation needs gene_selection mode 'all' or 'file' so that all sites select the same genes")
    selected_genes_path = request.run_dir / "selected_genes.tsv"

    if extractor is None:
        extractor = pgen_extract
    sample_inputs = load_sample_inputs(
        phenotype_path=request.phenotype,
        covariate_path=request.covariate,
        ancestry_path=request.ancestry,
        phenotype_id_column=request.phenotype_id_column,
        phenotype_columns=request.phenotype_columns,
        covariate_id_column=request.covariate_id_column,
        covariate_column=request.covariate_column,
        covariate_columns=request.covariate_columns,
        is_cov_single_column=request.is_cov_single_column,
        ancestry_groups=request.ancestries,
        ancestry_id_column=request.ancestry_id_column,
        ancestry_column=request.ancestry_column,
        num_cov=request.num_cov,
    )

    if request.samples_per_cohort == 0:
        samples_per_cohort: int | Literal["all"] = "all"
    else:
        samples_per_cohort = request.samples_per_cohort

    file_genes: tuple[GeneRef, ...] = ()
    if request.gene_selection.mode == "file":
        if request.gene_selection.path is None:
            raise ValueError(
                "gene selection path is required in file mode"
            )
        file_genes = read_gene_panel(
            request.gene_selection.path
        )

    selected_genes = []
    rows_by_ancestry = {}
    print(
        f"Running preprocessing for "
        f"{len(request.chromosomes)} chromosomes and "
        f"{len(request.ancestries)} ancestries...",
        flush=True,
    )
    for index, chromosome in enumerate(request.chromosomes):
        inputs = load_inputs(
            pgen_prefix=chromosome_path(
                request.genotype,
                chromosome,
            ),
            gene_panel_path=chromosome_path(
                request.gene_panel,
                chromosome,
            ),
            annotation_path=chromosome_path(
                request.annotation,
                chromosome,
            ),
        )

        if index == 0:
            reference_chromosome = chromosome
            reference_psam_ids = inputs.psam_ids

            for ancestry in request.ancestries:
                rows_by_ancestry[ancestry] = select_rows(
                    psam_ids=reference_psam_ids,
                    phenotypes=sample_inputs.phenotypes,
                    covariates=sample_inputs.covariates,
                    ancestries=sample_inputs.ancestries,
                    ancestry=ancestry,
                    phenotype_columns=request.phenotype_columns,
                    samples_per_cohort=samples_per_cohort,
                    sample_seed=request.sample_seed,
                    party=request.party,
                    num_sites=request.num_sites,
                )
        elif inputs.psam_ids != reference_psam_ids:
            raise ValueError(
                f"ordered PSAM IDs differ for chromosome "
                f"{reference_chromosome} and chromosome {chromosome}"
            )

        chromosome_groups = select_gene_groups(
            request=request.gene_selection,
            inputs=inputs,
            chromosome=chromosome,
            mask=request.mask,
            max_maf=request.max_maf,
            file_genes=file_genes,
        )
        chromosome_genes = tuple(
            group.gene for group in chromosome_groups
        )
        selected_genes.extend(chromosome_genes)

        for ancestry in request.ancestries:
            rows = rows_by_ancestry[ancestry]
            build = partial(
                prepare_blocks,
                pgen_prefix=inputs.pgen_prefix,
                gene_variants=chromosome_groups,
                rows=rows,
                role_seed=request.role_seed,
                shared_rate=request.shared_rate,
                keep_zero_variants=request.keep_zero_variants,
                extractor=extractor,
            )
            if request.party:
                build = partial(
                    prepare_site_blocks, pgen_prefix=inputs.pgen_prefix,
                    gene_variants=chromosome_groups, rows=rows,
                    extractor=extractor, party=request.party,
                    keep_zero_variants=request.keep_zero_variants,
                )
            if request.prepared_cache_dir is not None:
                prepare_cached_blocks(request, chromosome, ancestry, build)
            else:
                directory = request.run_dir / "prepared" / ancestry / f"chr{chromosome}"
                build(out_dir=directory)
                (directory / "config.json").write_text(json.dumps(
                    prepared_cache_config(request, chromosome, ancestry), sort_keys=True, indent=2,
                ) + "\n")

    selected_genes = tuple(selected_genes)
    if request.gene_selection.mode != "all":
        write_selected_genes(
            path=selected_genes_path,
            genes=selected_genes,
        )
        print("Wrote selected genes to", selected_genes_path)
    return selected_genes

def read_prepare_request(
    stream: TextIO,
) -> PrepareRequest:
    payload = json.load(stream)
    gene_selection = payload["gene_selection"]
    gene_selection_path = gene_selection["path"]

    return PrepareRequest(
        run_dir=Path(payload["run_dir"]),
        party=payload.get("party", 0),
        num_sites=payload.get("num_sites", 2),
        keep_zero_variants=payload.get("keep_zero_variants", False),
        prepared_cache_dir=Path(payload["prepared_cache_dir"]) if payload.get("prepared_cache_dir") else None,
        chromosomes=tuple(payload["chromosomes"]),
        genotype=payload["genotype"],
        gene_panel=payload["gene_panel"],
        annotation=payload["annotation"],
        phenotype=Path(payload["phenotype"]),
        covariate=Path(payload["covariate"]),
        ancestry=Path(payload["ancestry"]),
        plink2_bin=payload["plink2_bin"],
        phenotype_id_column=payload["phenotype_id_column"],
        covariate_id_column=payload["covariate_id_column"],
        covariate_column=payload["covariate_column"],
        covariate_columns=tuple(payload["covariate_columns"] or ()),
        is_cov_single_column=payload["is_cov_single_column"],
        ancestry_id_column=payload["ancestry_id_column"],
        ancestry_column=payload["ancestry_column"],
        phenotype_columns=tuple(payload["phenotype_columns"]),
        ancestries=tuple(
            ancestry.strip().upper()
            for ancestry in payload["ancestries"]
        ),
        num_cov=payload["num_cov"],
        mask=payload["mask"],
        max_maf=payload["max_maf"],
        samples_per_cohort=payload["samples_per_cohort"],
        sample_seed=payload["sample_seed"],
        role_seed=payload["role_seed"],
        shared_rate=payload["shared_rate"],
        gene_selection=GeneSelectionRequest(
            mode=gene_selection["mode"],
            per_chromosome=gene_selection["per_chromosome"],
            seed=gene_selection["seed"],
            path=(
                Path(gene_selection_path)
                if gene_selection_path
                else None
            ),
        ),
    )


def main() -> None:
    request = read_prepare_request(sys.stdin)
    prepare_chromosomes(request)


if __name__ == "__main__":
    main()
