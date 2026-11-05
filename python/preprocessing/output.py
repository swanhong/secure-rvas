import csv
import json
from pathlib import Path
import numpy as np
from .model import GeneBlock, GeneRef, PhenoCovRows

def write_selected_genes(
        path: Path,
        genes: tuple[GeneRef, ...],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(
            file,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writerow(("gene_id", "gene_symbol", "chromosome", "order_index"))
        for gene in genes:
            writer.writerow((gene.gene_id, gene.gene_symbol, gene.chromosome, str(gene.order_index)))

def site_directory(party: int, num_sites: int) -> str:
    return "AB"[party - 1] if num_sites == 2 else f"P{party}"


def write_outputs(
    out_dir: Path,
    blocks: tuple[GeneBlock, ...],
    rows: tuple[PhenoCovRows, ...],
    party: int = 0,
) -> None:
    """Write gene-aligned keys and genotype columns; P0 has no data directory."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"out_dir is not empty: {out_dir}")
    sites = []
    for index, local in enumerate(rows):
        if party and index + 1 != party:
            continue
        name = site_directory(index + 1, len(rows))
        directory = out_dir / name
        geno_dir = directory / "geno"
        geno_dir.mkdir(parents=True, exist_ok=True)
        with (directory / "keys.tsv").open("w") as keys:
            for g, block in enumerate(blocks):
                np.asarray(block.genotypes[index], dtype=np.int8).tofile(geno_dir / f"block.{g}.bin")
                keys.write("\t".join(block.keys[index]) + "\n")
        (directory / "input.json").write_text(json.dumps([block.keys[index] for block in blocks]) + "\n")
        (directory / "samples.txt").write_text("".join(f"{sample}\n" for sample in local.sample_ids))
        np.savetxt(directory / "cov.txt", local.covariates, fmt="%.17g", delimiter="\t")
        np.savetxt(directory / "pheno.txt", local.phenotypes, fmt="%.17g", delimiter="\t")
        sites.append(dict(party=index + 1, directory=name, samples=len(local.sample_ids)))
    (out_dir / "genes.txt").write_text("".join(f"{block.gene.gene_id}\n" for block in blocks))
    (out_dir / "info.json").write_text(json.dumps(dict(
        format_version=4, num_sites=len(rows), sites=sites,
        genes=len(blocks), genotype_dtype="int8", genotype_order="sample-major",
    ), indent=2) + "\n")
