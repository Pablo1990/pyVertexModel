#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pyVertexModel/run_cyst_aspect_cell_height.py"
OUTPUT_ROOT="${PYVERTEXMODEL_OUTPUT_ROOT:-$PROJECT_DIR/Result/aspect_ratio_cell_height_ablation_sweep}"

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

case "${1:-${SWEEP_SPLIT:-all}}" in
    first|1)
        parameter_indices=($(seq 0 51))
        ;;
    second|2)
        parameter_indices=($(seq 52 104))
        ;;
    all)
        parameter_indices=($(seq 0 104))
        ;;
    *)
        echo "Usage: $0 [first|second|all]"
        echo "  first:  jobs 0-51  (52 runs)"
        echo "  second: jobs 52-104 (53 runs)"
        echo "  all:    jobs 0-104 (105 runs)"
        exit 2
        ;;
esac

max_jobs="${MAX_JOBS:-3}"

run_simulation() {
    local idx=$1
    local log_file="$OUTPUT_ROOT/logs/sweep_job_${idx}.log"

    echo "[$(date)] Starting sweep job $idx"

    "$PYTHON" "$PYTHON_SCRIPT" "$idx" \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        echo "[$(date)] Finished sweep job $idx"
    else
        echo "[$(date)] Sweep job $idx failed with code $exit_code"
    fi
}

for idx in "${parameter_indices[@]}"; do
    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
        sleep 1
    done

    run_simulation "$idx" &
done

wait
echo "[$(date)] All aspect-ratio/cell-height/ablation simulations completed."
