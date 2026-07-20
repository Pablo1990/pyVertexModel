#!/bin/bash

PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
PYTHON_SCRIPT="$PROJECT_DIR/run_cyst_aspect.py"

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export QT_QPA_PLATFORM="offscreen"
export PYVISTA_OFF_SCREEN="true"

parameter_indices=(0 1 2 3 4)
max_jobs=3

run_simulation() {
    local idx=$1
    local log_file="$PROJECT_DIR/aspect_ratio_${idx}.log"

    echo "[$(date)] Starting aspect-ratio index $idx"

    python "$PYTHON_SCRIPT" "$idx" \
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