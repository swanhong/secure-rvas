#!/usr/bin/env bash

SF_RVAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"
export PATH="$SF_RVAT_ROOT/.local/bin:$PATH"
if [[ -f "$SF_RVAT_ROOT/.venv/bin/activate" ]]; then
  source "$SF_RVAT_ROOT/.venv/bin/activate"
fi
if command -v Rscript >/dev/null 2>&1; then
  export LD_LIBRARY_PATH="$(Rscript -e 'cat(R.home("lib"))')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
