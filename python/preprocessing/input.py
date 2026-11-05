import csv
import math
from pathlib import Path

from .model import (
    AnnotationRef,
    GeneRef,
    VariantRef,
    PrepInputs,
    SampleInputs,
)

MISSING_VALUES = {"", "NA", "NaN", "nan", ".", None}


def read_pvar(
    pvar_path: Path,
) -> tuple[VariantRef, ...]:
    with pvar_path.open(newline="") as file:
        reader = csv.reader(
            file,
            delimiter="\t",
        )

        for header in reader:
            if header and header[0] == "#CHROM":
                break
        else:
            raise ValueError("PVAR header not found")

        position_column = header.index("POS")
        id_column = header.index("ID")
        filter_column = (
            header.index("FILTER") if "FILTER" in header else None
        )

        return tuple(
            VariantRef(
                key=row[id_column],
                position=int(row[position_column]),
                filter_value=(
                    "PASS" if filter_column is None else row[filter_column]
                ),
            )
            for row in reader
            if row
        )

def read_psam(
    psam_path: Path,
) -> tuple[str, ...]:
    with psam_path.open(newline="") as file:
        reader = csv.reader(
            file,
            delimiter="\t",
        )
        header = next(reader)
        header[0] = header[0].lstrip("#")
        iid_column = header.index("IID")

        return tuple(
            row[iid_column]
            for row in reader
            if row
        )

def read_gene_panel(
    panel_path: Path,
) -> tuple[GeneRef, ...]:
    with panel_path.open(newline="") as file:
        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        genes = []
        for line_number, row in enumerate(reader, start=2):
            if not row["gene_id"].strip():
                raise ValueError(
                    f"{panel_path}:{line_number}: gene_id is required"
                )
            genes.append(
                GeneRef(
                    gene_id=row["gene_id"],
                    gene_symbol=row["gene_symbol"],
                    chromosome=row["chromosome"],
                    order_index=int(row["order_index"]),
                )
            )
        return tuple(genes)

def read_annotations(
    annotation_path: Path,
) -> tuple[
    tuple[AnnotationRef, ...],
    tuple[str, ...],
]:
    with annotation_path.open(newline="") as file:
        reader = csv.DictReader(
            file,
            delimiter="\t",
        )
        annotation_columns = tuple(
            reader.fieldnames[3:]
        )

        annotations = tuple(
            AnnotationRef(
                variant_key=row["variant_key"],
                gene_id=row["gene_id"],
                gene_symbol=row["gene_symbol"],
                values={
                    column: row[column]
                    for column in annotation_columns
                },
            )
            for row in reader
        )

        return annotations, annotation_columns

def read_phenotype_table(
    table_path: Path,
    delimiter: str,
    id_column: str,
    value_columns: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    with table_path.open(newline="") as file:
        reader = csv.DictReader(
            file,
            delimiter=delimiter,
        )
        if not reader.fieldnames:
            raise ValueError(f"no header in {table_path}")

        required_columns = (id_column, *value_columns)
        missing_columns = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"missing columns in {table_path}: "
                f"{', '.join(missing_columns)}"
            )

        return {
            row[id_column]: {
                column: (
                    float("nan")
                    if row[column] in MISSING_VALUES
                    else float(row[column])
                )
                for column in value_columns
            }
            for row in reader
        }


def read_covariate_table(
    table_path: Path,
    delimiter: str,
    id_column: str,
    covariate_column: str,
    num_cov: int,
) -> dict[str, tuple[float, ...]]:
    with table_path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"no header in {table_path}")
        for column in (id_column, covariate_column):
            if column not in reader.fieldnames:
                raise ValueError(f"missing {column} column in {table_path}")

        covariates = {}
        for row in reader:
            encoded = row[covariate_column].strip()
            if not encoded.startswith("[") or not encoded.endswith("]"):
                raise ValueError(
                    f"invalid {covariate_column} value for {row[id_column]}"
                )

            contents = encoded[1:-1]
            fields = contents.split(",") if contents else []
            if len(fields) < num_cov:
                raise ValueError(
                    f"{covariate_column} for {row[id_column]} has "
                    f"{len(fields)} values, need {num_cov}"
                )

            values = tuple(float(value.strip()) for value in fields[:num_cov])
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"non-finite {covariate_column} value for {row[id_column]}"
                )
            covariates[row[id_column]] = values

        return covariates


def read_ancestry_table(
    table_path: Path,
    delimiter: str,
    id_column: str,
    ancestry_column: str,
) -> dict[str, str]:
    with table_path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"no header in {table_path}")
        for column in (id_column, ancestry_column):
            if column not in reader.fieldnames:
                raise ValueError(f"missing {column} column in {table_path}")

        return {
            row[id_column]: row[ancestry_column].strip().upper()
            for row in reader
        }


def load_sample_inputs(
    phenotype_path: Path,
    covariate_path: Path,
    ancestry_path: Path,
    phenotype_id_column: str,
    phenotype_columns: tuple[str, ...],
    covariate_id_column: str,
    covariate_column: str,
    covariate_columns: tuple[str, ...],
    is_cov_single_column: bool,
    ancestry_groups: tuple[str, ...],
    ancestry_id_column: str,
    ancestry_column: str,
    num_cov: int,
) -> SampleInputs:
    print("Loading sample inputs")
    print(f"  phenotype: {phenotype_path}")
    print(f"  covariate: {covariate_path}")
    print(f"  ancestry: {ancestry_path}")
    covariates = {}
    for ancestry in ancestry_groups if "{anc}" in str(covariate_path) else ("",):
        path = Path(str(covariate_path).replace("{anc}", ancestry.lower()))
        if is_cov_single_column:
            table = read_covariate_table(
                path, "\t", covariate_id_column, covariate_column, num_cov,
            )
        else:
            rows = read_phenotype_table(
                path, "\t", covariate_id_column, covariate_columns,
            )
            table = {
                sample_id: tuple(values[column] for column in covariate_columns)
                for sample_id, values in rows.items()
            }
        covariates.update(table)
    return SampleInputs(
        phenotypes=read_phenotype_table(
            phenotype_path,
            delimiter=",",
            id_column=phenotype_id_column,
            value_columns=phenotype_columns,
        ),
        covariates=covariates,
        ancestries=read_ancestry_table(
            ancestry_path,
            delimiter="\t",
            id_column=ancestry_id_column,
            ancestry_column=ancestry_column,
        ),
    )

def load_inputs(
    pgen_prefix: Path,
    gene_panel_path: Path,
    annotation_path: Path,
) -> PrepInputs:
    pvar_path = Path(f"{pgen_prefix}.pvar")
    psam_path = Path(f"{pgen_prefix}.psam")

    annotations, annotation_columns = read_annotations(
        annotation_path
    )

    return PrepInputs(
        pgen_prefix=pgen_prefix,
        psam_ids=read_psam(psam_path),
        gene_panel=read_gene_panel(gene_panel_path),
        variants=read_pvar(pvar_path),
        annotations=annotations,
        annotation_columns=annotation_columns,
    )
