#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main_paper_simulations.py"

# Define the range of num_parameters to use
num_parameters=$(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)

# Function to run a simulation
run_simulation() {
    local num_parameter=$1
    echo "Running simulation for num_parameter: $num_parameter"
    python $PYTHON_SCRIPT "$num_parameter"
    echo "Finished simulation for num_parameter: $num_parameter"
}

# Run the simulations in parallel with a limit of 5 at a time
max_jobs=5
for num_parameter in $num_parameters; do
    ((i=i%max_jobs)); ((i++==0)) && wait
    run_simulation "$num_parameter" &
done
wait
