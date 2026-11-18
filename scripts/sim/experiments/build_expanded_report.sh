#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STUDY="$ROOT/log/optimization_navigation_20261013_v2"
REPORT="$ROOT/docs/reports/terrain_control_expansion_20261013_v2"
BUILD="$ROOT/log/latex_expansion_20261013_v2"
cd "$ROOT"
if [[ "${1:-}" != "--reuse-assets" ]]; then
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-experiments/bin/python \
    scripts/sim/experiments/audit_paper_campaign.py --expanded \
    --dataset "$STUDY/holdout_clouds/evaluation_manifest.json" \
    --results "$STUDY/holdout_final/results.json" --core-source "$STUDY/frozen_final" \
    --output "$REPORT/assets/audit.json"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-experiments/bin/python \
    scripts/sim/experiments/audit_paper_campaign.py --expanded --completion \
    --dataset "$STUDY/completion_catalog.json" \
    --results "$STUDY/completion_regression/results.json" \
    --output "$REPORT/assets/completion_audit.json"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-experiments/bin/python \
    scripts/sim/experiments/plot_expanded_campaign.py \
    --dataset "$STUDY/holdout_clouds/evaluation_manifest.json" \
    --terrain-results "$STUDY/holdout_final/results.json" \
    --navigation "$STUDY/navigation_holdout_h" \
    --isaac "$STUDY/isaac_final_j/manifest.json" \
    --isaac-candidate "$STUDY/isaac_semantics_k/manifest.json" \
    --completion "$STUDY/completion_regression/results.json" \
    --completion-audit "$REPORT/assets/completion_audit.json" \
    --capture "$STUDY/isaac_capture_m/manifest.json" \
    --safety "$STUDY/navigation_safety_i_verified" --output "$REPORT/assets"
fi
mkdir -p "$BUILD"
cd "$REPORT"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$BUILD" -jobname=PTR_Nav_Terrain_Control_Expansion_20261013 main.tex
cp "$BUILD/PTR_Nav_Terrain_Control_Expansion_20261013.pdf" "$REPORT/"
printf 'Report: %s\n' "$REPORT/PTR_Nav_Terrain_Control_Expansion_20261013.pdf"
