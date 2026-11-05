#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup/env.sh

# One config contains preparation settings and configLocal.Party0..N.toml.
config="${CONFIG:-config/1kg-nparty}"
reference_engine="${REFERENCE_ENGINE:-python}"

step_labels=()
step_seconds=()
run_step() {
  local label="$1"
  shift
  echo "$label"
  local started_at="$SECONDS"
  "$@"
  step_labels+=("$label")
  step_seconds+=("$((SECONDS - started_at))")
}

mod_flag=""
if [[ -d vendor ]]; then
  mod_flag="-mod=vendor"
fi
go build $mod_flag -o secure-rvas .
num_pheno="$(python3 -c 'import sys,tomllib; print(len(tomllib.load(open(sys.argv[1], "rb"))["phenotype_columns"]))' "$config/configGlobal.toml")"

run_step "[0/7] Generate 1000 Genomes test data" \
  python3 python/datasets/onekg/prepare_1kgenome.py --config "$config" --num-pheno "$num_pheno"

run_step "[1/7] Prepare N sites (shared and site-only variants)" \
  ./secure-rvas prepare --config "$config"

run_step "[2/7] Generate shared PRG keys" \
  ./secure-rvas keygen --config "$config"

run_step "[3/7] Run secure Burden/SKAT with P0 and all N data sites" \
  ./secure-rvas run --config "$config"

run_step "[4/7] Run $reference_engine reference on all N sites pooled by variant key" \
  python3 python/analysis/run_reference.py --config "$config" --engine "$reference_engine"

run_step "[5/7] Compare secure and reference results" \
  python3 python/analysis/compare_secure_to_reference.py --config "$config"

run_step "[6/7] Generate scatter and Manhattan plots" \
  python3 python/analysis/plot_secure_vs_reference.py --config "$config"

run_step "[7/7] Summarize metrics" \
  "${PYTHON_BIN:-python3}" python/analysis/summarize_metrics.py --config "$config" --all-parties

echo "Secure RVAS N-party 1KG workflow completed"
echo "Workflow timing summary"
for index in "${!step_labels[@]}"; do
  printf "  %-80s %d seconds\n" "${step_labels[index]}" "${step_seconds[index]}"
done
