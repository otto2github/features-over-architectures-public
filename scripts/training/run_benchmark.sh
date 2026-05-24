#!/usr/bin/env bash
# =============================================================================
# Features-over-Architectures benchmark driver
# =============================================================================
# This script re-runs the benchmark cells reported in the paper, organised
# into self-contained phases so that any phase can be re-run in isolation.
#
# Usage:
#   bash run_benchmark.sh smoke         # ~5 min smoke test (pipeline check)
#   bash run_benchmark.sh prep          # one-off: build node_features_v1_1.parquet
#   bash run_benchmark.sh main          # main benchmark (15 cells, ~3.5 h on a single GPU)
#   bash run_benchmark.sh ablation      # single-modality ablation (12 cells, ~2.5 h)
#   bash run_benchmark.sh robustness    # three-label robustness sweep (9 cells, ~2 h)
#   bash run_benchmark.sh all           # prep + smoke + main + ablation + robustness
#
# Environment variables:
#   PROJECT_ROOT   path containing data/processed/kg/, defaults to ~/code/foa_project
#   EPOCHS         training epochs per cell, default 100 (smoke phase forces 10)
#   HIDDEN_DIM     hidden dimension, default 64
# =============================================================================
set -euo pipefail

PHASE="${1:-help}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/code/foa_project}"
EPOCHS="${EPOCHS:-100}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"

if command -v uv >/dev/null 2>&1 && [ -d "${PROJECT_ROOT}" ]; then
  PYRUN=(uv run --project "${PROJECT_ROOT}/.venv" python)
else
  PYRUN=(python3)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
  cat <<EOF
Usage:
  bash run_benchmark.sh <phase>

Phases:
  smoke        ~5 min smoke test
  prep         one-off node_features_v1_1.parquet preparation
  main         main benchmark (15 cells)
  ablation     single-modality ablation (12 cells)
  robustness   three-label robustness sweep (9 cells)
  all          prep + smoke + main + ablation + robustness

Environment overrides:
  PROJECT_ROOT=${PROJECT_ROOT}
  EPOCHS=${EPOCHS}
  HIDDEN_DIM=${HIDDEN_DIM}
EOF
}

run_prep() {
  echo "==> [prep] Building node_features_v1_1.parquet"
  "${PYRUN[@]}" prepare_node_features_v1_1.py --project-root "${PROJECT_ROOT}"
}

run_smoke() {
  echo "==> [smoke] One MLP × M11 cell with epochs=10"
  "${PYRUN[@]}" gnn_baseline_v1_1.py \
    --model MLP --modal M11 --seed 42 --epochs 10 --hidden-dim "${HIDDEN_DIM}"
}

run_main() {
  echo "==> [main] 5 models × 3 modalities × 5 seeds = 75 cells"
  for MODEL in MLP GCN GAT GraphSAGE RGCN; do
    for MODAL in M5 M10 M11; do
      for SEED in 42 123 456 789 1024; do
        echo "    main: ${MODEL} × ${MODAL} × seed=${SEED}"
        "${PYRUN[@]}" gnn_baseline_v1_1.py \
          --model "${MODEL}" --modal "${MODAL}" --seed "${SEED}" \
          --epochs "${EPOCHS}" --hidden-dim "${HIDDEN_DIM}"
      done
    done
  done
}

run_ablation() {
  echo "==> [ablation] single-modality contribution (M6/M7/M8/M9)"
  for MODEL in MLP GCN GAT GraphSAGE RGCN; do
    for MODAL in M6 M7 M8 M9; do
      for SEED in 42 123 456 789 1024; do
        echo "    ablation: ${MODEL} × ${MODAL} × seed=${SEED}"
        "${PYRUN[@]}" gnn_baseline_v1_1.py \
          --model "${MODEL}" --modal "${MODAL}" --seed "${SEED}" \
          --epochs "${EPOCHS}" --hidden-dim "${HIDDEN_DIM}"
      done
    done
  done
}

run_robustness() {
  echo "==> [robustness] M11 × 3 label protocols × 5 seeds"
  for LABEL in fraud_v07 fraud_v08_strict fraud_v08_loose; do
    for MODEL in MLP GCN GAT GraphSAGE RGCN; do
      for SEED in 42 123 456 789 1024; do
        echo "    robustness: ${MODEL} × M11 × ${LABEL} × seed=${SEED}"
        "${PYRUN[@]}" gnn_baseline_v1_1.py \
          --model "${MODEL}" --modal M11 --seed "${SEED}" \
          --label-col "${LABEL}" \
          --epochs "${EPOCHS}" --hidden-dim "${HIDDEN_DIM}"
      done
    done
  done
}

case "${PHASE}" in
  help|-h|--help)  show_help ;;
  prep)            run_prep ;;
  smoke)           run_smoke ;;
  main)            run_main ;;
  ablation)        run_ablation ;;
  robustness)      run_robustness ;;
  all)             run_prep; run_smoke; run_main; run_ablation; run_robustness ;;
  *)
    echo "Unknown phase: ${PHASE}"
    show_help
    exit 1
    ;;
esac

echo "==> Done: ${PHASE}"
