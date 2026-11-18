#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPORT="$ROOT/docs/reports/thesis_evidence_20261014"
BUILD="$ROOT/log/latex_thesis_evidence_20261014"
cd "$ROOT"
if [[ "${1:-}" != "--reuse-assets" ]]; then
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-experiments/bin/python \
    scripts/sim/experiments/plot_thesis_evidence.py \
    --analytic log/thesis_evidence_20261014/analytic_yaw_frozen/results.json \
    --output "$REPORT/assets"
fi
mkdir -p "$BUILD"
cd "$REPORT"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$BUILD" -jobname=PTR_Nav_Thesis_Evidence_20261014 main.tex
cp "$BUILD/PTR_Nav_Thesis_Evidence_20261014.pdf" "$REPORT/"
printf 'Report: %s\n' "$REPORT/PTR_Nav_Thesis_Evidence_20261014.pdf"
