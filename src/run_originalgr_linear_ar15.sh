#!/bin/bash -l

#$ -l h_rt=24:0:0
#$ -l mem=5G
#$ -l tmpfs=10G
#$ -N optuna_originalgr_ar15
#$ -pe smp 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_DIR="$PROJECT_DIR/Result/ellipsoid_calibration_originalgr_linear"

if [ -n "${JOB_ID:-}" ] && [ -n "${TMPDIR:-}" ]; then
    RUN_DIR="$TMPDIR/pyVertexModel_originalgr_ar15"
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR"
else
    cd "$PROJECT_DIR"
fi

if [ -x "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export QT_QPA_PLATFORM="offscreen"
export PYVISTA_OFF_SCREEN="true"

mkdir -p "$RESULT_DIR/logs"

LOG_FILE="$RESULT_DIR/logs/originalgr_ar15_${JOB_ID:-local}.log"

echo "[$(date)] Running original-gradient AR 1.5 Optuna diagnostic"
"$PYTHON" -m pyVertexModel.main_optuna_ellipsoid_calibration_originalgr \
    --aspect-ratio 1.5 \
    --trials 100 \
    > "$LOG_FILE" 2>&1
echo "[$(date)] Done. Log: $LOG_FILE"
