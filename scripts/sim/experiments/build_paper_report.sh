#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPORT="$ROOT/docs/reports/terrain_navigation_20261012"
BUILD="$ROOT/log/latex_report_20261012"
if [[ "${1:-}" != "--reuse-assets" ]]; then
  cd "$ROOT"
  PYTHON="$ROOT/.venv-experiments/bin/python"
  [[ -x "$PYTHON" ]] || PYTHON=python3
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" \
    scripts/sim/experiments/prepare_paper_assets.py --output "$REPORT/assets"
  if [[ -f "$ROOT/log/public_cloud_bgk_disjoint_20261012/results.json" ]]; then
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" \
      scripts/sim/experiments/plot_public_clouds.py \
      --dataset log/public_cloud_dataset_20261012/dataset_manifest.json \
      --batches log/public_cloud_gp_20261012/batch_manifest.json \
                log/public_cloud_gp_disjoint_20261012/batch_manifest.json \
      --risk-results log/public_cloud_bgk_audit_20261012/results.json \
                     log/public_cloud_bgk_disjoint_20261012/results.json \
      --output "$REPORT/assets"
  fi
fi
mkdir -p "$BUILD"
cd "$REPORT"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$BUILD" -jobname=PTR_Nav_Experiment_Report_20261012 main.tex
cp "$BUILD/PTR_Nav_Experiment_Report_20261012.pdf" "$REPORT/"
printf 'Report: %s\n' "$REPORT/PTR_Nav_Experiment_Report_20261012.pdf"
