#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main_paper_simulations.py"
OUTPUT_DIR="Result/to_calculate_ps_recoil/c/"

# Define the project directory and add it to PYTHONPATH
PROJECT_DIR=$(dirname "$(dirname "$(realpath $0)")")
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export QT_QPA_PLATFORM="offscreen"

# Parameters to run
#num_parameters=(0 1 2 3 4)
num_parameters=(0)

# Function to run a simulation
run_simulation() {
    local num_parameter=$1
    local dir_name=$2
    echo $OUTPUT_DIR
    echo "Running simulation number $num_parameter"
    python $PYTHON_SCRIPT "$num_parameter" "$OUTPUT_DIR/$dir_name"
    echo "Finished simulation number $num_parameter"
}

# Parallel execution
max_jobs=1
for num_parameter in "${num_parameters[@]}"; do
    for dir in "$PROJECT_DIR/$OUTPUT_DIR"/* ; do
        echo "$dir"
        dir_name=$(basename "$dir")
        while [ $(jobs -r | wc -l) -ge "$max_jobs" ]; do
            sleep 1
        done
        run_simulation "$num_parameter" "$dir_name" &
    done
done

wait
echo "[$(date)] All simulations completed."