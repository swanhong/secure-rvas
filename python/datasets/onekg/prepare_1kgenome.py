from __future__ import annotations

import argparse
from pathlib import Path
import tomllib
import utils


VCF_TEMPLATE = (
    "1kGP_high_coverage_Illumina.chr{chromosome}."
    "filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)
VCF_BASE_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/"
    "data_collections/1000G_2504_high_coverage/working/"
    "20220422_3202_phased_SNV_INDEL_SV/"
)
PANEL_NAME = "integrated_call_samples_v3.20130502.ALL.panel"
PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/"
    f"release/20130502/{PANEL_NAME}"
)
GTF_NAME = "gencode.v50.annotation.gtf.gz"
GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/"
    f"Gencode_human/release_50/{GTF_NAME}"
)


def parse_chromosomes(values: list[str]) -> list[int]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(range(1, 23))

    chromosomes = []
    seen = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                raise ValueError("empty chromosome value")

            if "-" in item:
                fields = item.split("-")
                if len(fields) != 2:
                    raise ValueError(f"invalid chromosome range: {item}")
                start, end = (int(field) for field in fields)
                if start > end:
                    raise ValueError(f"chromosome range is descending: {item}")
                selected = range(start, end + 1)
            else:
                selected = (int(item),)

            for chromosome in selected:
                if chromosome not in range(1, 23):
                    raise ValueError(
                        f"chromosome must be between 1 and 22: {chromosome}"
                    )
                if chromosome in seen:
                    raise ValueError(f"duplicate chromosome: {chromosome}")
                seen.add(chromosome)
                chromosomes.append(chromosome)

    if not chromosomes:
        raise ValueError("at least one chromosome is required")
    return chromosomes


def read_psam_ids(path: Path) -> tuple[str, ...]:
    with path.open() as file:
        columns = next(file).lstrip("#").split()
        iid_column = columns.index("IID")
        return tuple(line.split()[iid_column] for line in file if line.strip())


def prepare_1kgenome(root: Path, args) -> None:
    chromosomes = parse_chromosomes(args.chromosome)
    generated = root / "generated"
    raw = generated / "raw"
    panel = utils.download_if_missing(PANEL_URL, raw / PANEL_NAME)
    gtf = utils.download_if_missing(GTF_URL, raw / GTF_NAME)

    psam_ids = ()
    pgen_prefixes = []
    for index, chromosome in enumerate(chromosomes):
        vcf_name = VCF_TEMPLATE.format(chromosome=chromosome)
        vcf = utils.download_if_missing(VCF_BASE_URL + vcf_name, raw / vcf_name)
        prefix = utils.create_pgen(
            vcf_path=vcf,
            panel_path=panel,
            out_prefix=generated / "genotype" / f"chr{chromosome}",
            keep_path=generated / "work" / "phase3.keep",
        )
        pgen_prefixes.append(prefix)
        frequency = utils.create_allele_frequencies(
            pgen_prefix=prefix,
            out_prefix=generated / "genotype" / f"chr{chromosome}",
        )
        current_ids = read_psam_ids(prefix.with_suffix(".psam"))
        if index == 0:
            psam_ids = current_ids
        elif current_ids != psam_ids:
            raise ValueError(
                f"ordered PSAM IDs differ for chr{chromosomes[0]} and chr{chromosome}"
            )

        gene_panel = generated / "gene_panel" / f"chr{chromosome}.tsv"
        annotation = generated / "annotation" / f"chr{chromosome}.tsv"
        utils.create_inputs_from_gencode(
            gtf_path=gtf,
            pvar_path=prefix.with_suffix(".pvar"),
            gene_panel_path=gene_panel,
            annotation_path=annotation,
            chromosome=str(chromosome),
            frequency_path=frequency,
        )

    chromosome_set = "chr" + "_".join(str(value) for value in chromosomes)
    eigenvec = utils.create_pca(
        pgen_prefixes=pgen_prefixes,
        work_dir=generated / "work" / "pca" / chromosome_set,
    )
    utils.create_ancestry_table(
        panel_path=panel,
        eigenvec_path=eigenvec,
        sample_ids=psam_ids,
        out_path=generated / "ancestry_pred.tsv",
    )
    sample_path = pgen_prefixes[0].with_suffix(".psam")
    utils.create_phenotype(
        sample_path,
        generated / "phenotype.csv",
        num_pheno=args.num_pheno,
        seed=42,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--num-pheno",
        type=int,
        default=1,
        help="number of phenotypes to generate",
    )
    args = parser.parse_args()

    with (args.config / "configGlobal.toml").open("rb") as config_file:
        config = tomllib.load(config_file)
    args.chromosome = [
        str(chromosome)
        for chromosome in config["chromosomes"]
    ]

    if args.num_pheno < 1:
        raise ValueError("num-pheno must be at least 1")
    prepare_1kgenome(Path(__file__).resolve().parents[3] / "data" / "1kg", args)


if __name__ == "__main__":
    main()
