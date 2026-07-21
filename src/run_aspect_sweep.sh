#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pyVertexModel/run_cyst_aspect.py"
OUTPUT_ROOT="${PYVERTEXMODEL_OUTPUT_ROOT:-$PROJECT_DIR/Result/aspect_ratio_sweep}"

if [ -x "$PROJECT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYVERTEXMODEL_OUTPUT_ROOT="$OUTPUT_ROOT"
export QT_QPA_PLATFORM="offscreen"
export PYVISTA_OFF_SCREEN="true"

mkdir -p "$OUTPUT_ROOT/logs"

parameter_indices=(0 1 2 3 4)
max_jobs=3

run_simulation() {
    local idx=$1
    local log_file="$OUTPUT_ROOT/logs/aspect_ratio_${idx}.log"

    echo "[$(date)] Starting aspect-ratio index $idx"

    "$PYTHON" "$PYTHON_SCRIPT" "$idx" \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date)] Finished aspect-ratio index $idx"
    else
        echo "[$(date)] Aspect-ratio index $idx failed with code $exit_code"
    fi
}

for idx in "${parameter_indices[@]}"; do
    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 1
    done

    run_simulation "$idx" &
done

wait
echo "[$(date)] All aspect-ratio simulations completed."
