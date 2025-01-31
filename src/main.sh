#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main.py"

# Define the project directory and add it to PYTHONPATH
PROJECT_DIR=$(dirname "$(dirname "$(realpath $0)")")
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Function to run a simulation
run_simulation() {
    local num_parameter=$1
    echo "Running simulation number $num_parameter"
    python $PYTHON_SCRIPT "$num_parameter" outputs/
    echo "Finished simulation number $num_parameter"
}

# Run the simulations in parallel with a limit of 5 at a time
max_jobs=1
for num_parameter in $(seq 1 $max_jobs); do
    run_simulation "$num_parameter" &
    sleep 10s
done
wait