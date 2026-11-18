#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPORT="$ROOT/docs/reports/systematic_study"
BUILD="$ROOT/log/latex_systematic_study"
OPTIONS=()
ANALYZER=systematic_study.py
SOURCE=main.tex
JOB=PTR_Nav_Systematic_Study
case "${1:-}" in
  "") ;;
  --reuse-assets) OPTIONS+=(--verify-assets) ;;
  --correction)
    ANALYZER=summarize_terrain_correction.py
    SOURCE=terrain_correction.tex
    JOB=PTR_Nav_Terrain_Correction
    BUILD="$ROOT/log/latex_terrain_correction"
    ;;
  --system-study)
    ANALYZER=report_terrain_system.py
    SOURCE=terrain_system_study.tex
    JOB=PTR_Nav_Terrain_System_Study
    BUILD="$ROOT/log/latex_terrain_system_study"
    ;;
  --comprehensive)
    ANALYZER=report_comprehensive.py
    REPORT="$ROOT/docs/reports/comprehensive_study"
    SOURCE=main.tex
    JOB=PTR_Nav_Comprehensive_Study
    BUILD="$ROOT/log/latex_comprehensive_study"
    ;;
  *) printf 'Usage: bash %s [--reuse-assets|--correction|--system-study|--comprehensive]\n' "$0" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  printf 'Only one optional argument is supported.\n' >&2
  exit 2
fi
cd "$ROOT"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-experiments/bin/python \
  "scripts/sim/experiments/$ANALYZER" "${OPTIONS[@]}"
mkdir -p "$BUILD"
cd "$REPORT"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$BUILD" -jobname="$JOB" "$SOURCE"
cp "$BUILD/$JOB.pdf" "$REPORT/"
printf 'Report: %s\n' "$REPORT/$JOB.pdf"
