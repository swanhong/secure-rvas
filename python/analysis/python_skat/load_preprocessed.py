from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlainInput:
    genotype_dirs: list[Path]
    genes: list[str]
    keys: list[list[list[str]]]
    sample_counts: list[int]
    covariates: np.ndarray
    phenotypes: np.ndarray


def read_text_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def read_keys(path: Path) -> list[list[str]]:
    """Line i holds gene i's variant keys separated by tabs, in block column order."""
    text = path.read_text()
    if not text:
        return []
    return [line.split("\t") if line else [] for line in text.removesuffix("\n").split("\n")]


def read_site_inputs(sites: list[dict[str, str]]) -> PlainInput:
    """Read one local site or pool the supplied data sites in order, excluding P0."""
    if not sites:
        raise ValueError("reference needs at least one data site")
    genes = Path(sites[0]["genes_file"]).read_text().split()
    keys, covariates, phenotypes, genotype_dirs = [], [], [], []
    for party, site in enumerate(sites, 1):
        if Path(site["genes_file"]).read_text().split() != genes:
            raise ValueError(f"P{party}: gene order differs")
        local_keys = read_keys(Path(site["keys_file"]))
        if len(local_keys) != len(genes):
            raise ValueError(f"P{party}: genes and key rows have different lengths")
        if any(len(row) != len(set(row)) for row in local_keys):
            raise ValueError(f"P{party}: duplicate key within a gene")
        x = read_text_matrix(Path(site["covariate_file"]))
        y = read_text_matrix(Path(site["phenotype_file"]))
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"P{party}: covariate and phenotype row counts differ")
        keys.append(local_keys)
        covariates.append(x)
        phenotypes.append(y)
        genotype_dirs.append(Path(site["genotype_dir"]))
    return PlainInput(
        genotype_dirs=genotype_dirs,
        genes=genes,
        keys=keys,
        sample_counts=[x.shape[0] for x in covariates],
        covariates=np.vstack(covariates),
        phenotypes=np.vstack(phenotypes),
    )


def read_plain_input(directory_a: Path, directory_b: Path | None = None) -> PlainInput:
    """Legacy A/B prepared-root interface."""
    roots = (directory_a, directory_a if directory_b is None else directory_b)
    return read_site_inputs([
        {
            "genes_file": str(root / "genes.txt"),
            "keys_file": str(root / party / "keys.tsv"),
            "covariate_file": str(root / party / "cov.txt"),
            "phenotype_file": str(root / party / "pheno.txt"),
            "genotype_dir": str(root / party / "geno"),
        }
        for root, party in zip(roots, ("A", "B"))
    ])


def read_dosages(path: Path, rows: int, columns: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.int8)
    return values.reshape(rows, columns).astype(np.float64)


def read_gene_genotype(input_data: PlainInput, gene_index: int) -> np.ndarray:
    """Union columns follow first appearance; unheld keys have dosage zero."""
    local_keys = [keys[gene_index] for keys in input_data.keys]
    column = {}
    for keys in local_keys:
        for key in keys:
            column.setdefault(key, len(column))
    genotype = np.zeros((sum(input_data.sample_counts), len(column)))
    offset = 0
    for directory, count, keys in zip(
        input_data.genotype_dirs, input_data.sample_counts, local_keys
    ):
        if keys:
            block = read_dosages(directory / f"block.{gene_index}.bin", count, len(keys))
            genotype[offset:offset + count, [column[key] for key in keys]] = block
        offset += count
    return genotype
