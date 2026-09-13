#!/bin/bash -l

#$ -l h_rt=24:0:0
#$ -l mem=5G
#$ -l tmpfs=10G
#$ -N optuna_line_tension_ar15
#$ -pe smp 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_DIR="$PROJECT_DIR/Result/ellipsoid_line_tension_calibration"

if [ -n "${JOB_ID:-}" ] && [ -n "${TMPDIR:-}" ]; then
    RUN_DIR="$TMPDIR/pyVertexModel_line_tension_ar15"
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR"
else
    cd "$PROJECT_DIR"
fi

if [ -x "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [ -x "$PROJECT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
else
    PYTHON="${PYTHON:-python}"
fi

export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="offscreen"
export PYVISTA_OFF_SCREEN="true"

mkdir -p "$RESULT_DIR/logs"

TRIALS="${TRIALS:-500}"
LOG_FILE="$RESULT_DIR/logs/line_tension_ar15_${JOB_ID:-local}.log"

echo "[$(date)] Running line-tension AR 1.5 Optuna calibration (${TRIALS} trials)"
"$PYTHON" -m pyVertexModel.main_optuna_ellipsoid_calibration_mockup \
    --aspect-ratio 1.5 \
    --trials "$TRIALS" \
    > "$LOG_FILE" 2>&1
echo "[$(date)] Done. Log: $LOG_FILE"
