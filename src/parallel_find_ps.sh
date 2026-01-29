#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/analysis/find_required_purse_string.py"

# Define the project directory and add it to PYTHONPATH
PROJECT_DIR=$(dirname "$(dirname "$(realpath $0)")")
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export QT_QPA_PLATFORM="offscreen"

# Function to run a simulation
run_simulation() {
    local num_parameter=$1
    echo "Running simulation number $num_parameter"
    python $PYTHON_SCRIPT "$num_parameter"
    echo "Finished simulation number $num_parameter"
}

# Get the number of folders in the specified directory
TARGET_DIR="Result/to_calculate_ps_recoil/c/"
num_folders=$(find "$PROJECT_DIR/$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Number of folders in $TARGET_DIR: $num_folders"

# Create an array of parameter indices based on the number of folders
num_parameters=()
for ((i=0; i<num_folders; i++)); do
    num_parameters+=("$i")
done

# Run the simulations in parallel with a limit of 5 at a time
max_jobs=2
for num_parameter in "${num_parameters[@]}"; do
    while [ $(jobs -r | wc -l) -ge "$max_jobs" ]; do
        sleep 1
    done
    run_simulation "$num_parameter" &
    sleep 10
done
wait