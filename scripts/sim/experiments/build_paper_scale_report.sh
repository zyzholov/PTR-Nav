#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPORT="$ROOT/docs/reports/paper_scale_replacements_20261013"
BUILD="$ROOT/log/latex_paper_scale_20261013"
cd "$ROOT"
if [[ "${1:-}" != "--reuse-assets" ]]; then
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$ROOT/.venv-experiments/bin/python" \
    scripts/sim/experiments/audit_paper_campaign.py \
    --dataset log/paper_replacement_clouds_20261013/dataset_manifest.json \
    --results log/paper_scale_results_20261013/results.json \
    --output "$REPORT/assets/audit.json"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$ROOT/.venv-experiments/bin/python" \
    scripts/sim/experiments/plot_paper_scale.py \
    --dataset log/paper_replacement_clouds_20261013/dataset_manifest.json \
    --results log/paper_scale_results_20261013/results.json --output "$REPORT/assets"
fi
mkdir -p "$BUILD"
cd "$REPORT"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$BUILD" -jobname=PTR_Nav_Five_Category_Study_20261013 main.tex
cp "$BUILD/PTR_Nav_Five_Category_Study_20261013.pdf" "$REPORT/"
printf 'Report: %s\n' "$REPORT/PTR_Nav_Five_Category_Study_20261013.pdf"
