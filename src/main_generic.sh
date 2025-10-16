#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main_optuna.py"

# Define the project directory and add it to PYTHONPATH
PROJECT_DIR=$(dirname "$(dirname "$(realpath $0)")")
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Function to run a simulation
run_simulation() {
    echo "Running simulation"
    python $PYTHON_SCRIPT
    echo "Finished simulation"
}

# Run the simulation
run_simulation &
wait