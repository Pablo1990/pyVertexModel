#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main_different_cell_shapes.py"

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

# Run the simulations in parallel with a limit of 5 at a time
num_parameters=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29) # (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
max_jobs=3
for num_parameter in "${num_parameters[@]}"; do
    while [ $(jobs -r | wc -l) -ge "$max_jobs" ]; do
        sleep 1
    done
    run_simulation "$num_parameter" &
done
wait