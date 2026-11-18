#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPORT="$ROOT/docs/reports/terrain_optimization_20261013"
BUILD="$ROOT/log/latex_optimization_20261013"
cd "$ROOT"
if [[ "${1:-}" != "--reuse-assets" ]]; then
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$ROOT/.venv-experiments/bin/python" \
    scripts/sim/experiments/summarize_optimization.py --output "$REPORT/assets"
fi
mkdir -p "$BUILD"
cd "$REPORT"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$BUILD" -jobname=PTR_Nav_Terrain_Optimization_20261013 main.tex
cp "$BUILD/PTR_Nav_Terrain_Optimization_20261013.pdf" "$REPORT/"
printf 'Report: %s\n' "$REPORT/PTR_Nav_Terrain_Optimization_20261013.pdf"
