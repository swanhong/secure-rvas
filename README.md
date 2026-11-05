# SF-RVAT

Code for **SF-RVAT: Secure Federated Rare-Variant Association Testing**.
Built on [SF-GWAS](https://github.com/hhcho/sfgwas); see [LICENSE](LICENSE).

## Data availability

The experiments in our paper use data from the [All of Us Research Program](https://www.researchallofus.org/data-tools/data-access/), which require authorized access and are not distributed with this repository. To enable public testing, we provide an end-to-end example using publicly available [1000 Genomes Project](https://www.internationalgenome.org/1000-genomes-summary/) data with synthetic phenotypes and variant annotations. This example demonstrates the workflow; it does not reproduce the paper’s All of Us results.

## Install

Install Go 1.21 or later, Python 3.11 or later (with `venv` and development headers), R (with development tools), a C/C++ compiler, and `curl`, then run from the repository root:

```bash
bash scripts/setup/install.sh
source scripts/setup/env.sh
go build -o secure-rvas .
```

The installer requires Linux x86-64 with AVX2 support. It creates `.venv`,
installs the Python dependencies, downloads
PLINK 2 into `.local/bin`, and installs the R package SKAT if needed.
The Python reference uses SKAT's Davies routine.

## 1KG test

```bash
bash scripts/run_1kg_np_workflow.sh
```

This runs P0 and three data sites on localhost using `config/1kg-nparty`:
download public 1000 Genomes and GENCODE inputs, prepare the data, generate
keys, run secure Burden/SKAT, and compare with the pooled plaintext reference.
Results and plots are written under `output/1kg-nparty`.

The default configuration produces 200 result rows (100 genes × two
phenotypes). In the validated example, 66 genes have testable pooled variants,
giving 132 rows for comparison with the plaintext reference. The remaining
34 genes have reference values of `NA` and are excluded from comparison metrics
and plots. This is expected behavior. The `failed` count in the Davies summary
refers to reference calculations flagged as non-converged, not secure execution
failures.

The example selects 100 genes on chromosome 22 from the EUR samples and uses
synthetic phenotypes and synthetic LoF labels.
Downloads require approximately 550 MB; allow several GB for generated files.

`config/aou-nparty` is a template for locally supplied AoU inputs.
