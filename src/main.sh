#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="pyVertexModel/main.py"

# Define the project directory and add it to PYTHONPATH
PROJECT_DIR=$(dirname "$(dirname "$(realpath $0)")")
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

python $PYTHON_SCRIPT
wait