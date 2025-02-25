#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main_paper_simulations.py"

# Define output directory
OUTPUT_DIR="Result/final_results"

# Define the project directory and add it to PYTHONPATH
PROJECT_DIR=$(dirname "$(dirname "$(realpath $0)")")
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Set the QT_QPA_PLATFORM environment variable to offscreen
export QT_QPA_PLATFORM=offscreen

# Define an array of parameters to run the simulation with
#num_parameters=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
num_parameters=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)

# Function to run a simulation
run_simulation() {
    local num_parameter=$1
    echo "Running simulation for num_parameter: $num_parameter"
    python $PYTHON_SCRIPT "$num_parameter" "$OUTPUT_DIR" #"120_mins"
    echo "Finished simulation for num_parameter: $num_parameter"
}

# Run the simulations in parallel with a limit of 5 at a time
max_jobs=4
for num_parameter in "${num_parameters[@]}"; do
    ((i=i%max_jobs)); ((i++==0)) && wait
    run_simulation "$num_parameter" &
done
wait
