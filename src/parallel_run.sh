#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main_paper_simulations.py"
OUTPUT_DIR="Result/final_results_wing_disc_real_bottom_right"

# Conda environment name (update this)
CONDA_ENV="pyVertexModel"

# Project directory setup
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")

# Parameters to run
num_parameters=(0 1 2 3 4 5 6 7 8 9)

run_simulation() {
    local num_parameter=$1
    echo "Running simulation for num_parameter: $num_parameter"
    #mamba run -n "$CONDA_ENV" nice -n -10 python "$PYTHON_SCRIPT" "$num_parameter" "$OUTPUT_DIR"
    sudo nice -n -15 bash -c "
        source '$(conda info --base)/etc/profile.d/conda.sh'
        conda activate '$CONDA_ENV'
        export PYTHONPATH='$PROJECT_DIR:$PYTHONPATH'
        export QT_QPA_PLATFORM='offscreen'
        python '$PYTHON_SCRIPT' '$num_parameter' '$OUTPUT_DIR'
    "
    echo "Finished simulation for num_parameter: $num_parameter"
}

# Parallel execution (max 2 jobs)
max_jobs=2
for num_parameter in "${num_parameters[@]}"; do
    while [ $(jobs -r | wc -l) -ge "$max_jobs" ]; do
        sleep 1
    done
    run_simulation "$num_parameter" &
done

wait
echo "[$(date)] All simulations completed."