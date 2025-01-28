#!/bin/bash -l

# Example batch script to run a Python script in a virtual environment.
# Adapted from https://github.com/UCL-ARC/myriad-python-analysis-example/blob/main/run_analysis.sh

# Request 1 minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:1:0

# Request 1 gigabyte of RAM for each core/thread
# (must be an integer followed by M, G, or T)
#$ -l mem=1G

# Request 1 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=1G

# Set the name of the job.
#$ -N pyVertexModel

# Request 1 cores.
#$ -pe smp 1

# Set the working directory to project directory in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /home/<your_UCL_id>/Scratch/pyVertexModel

# Load python3 module - this must be the same version as loaded when creating and
# installing dependencies in the virtual environment
module load python3/3.9

# Define a local variable pointing to the project directory in your scratch space
PROJECT_DIR=/home/<your_UCL_id>/Scratch/pyVertexModel

# Activate the virtual environment in which you installed the project dependencies
source $PROJECT_DIR/venv/bin/activate

# Change current working directory to temporary file system on node
cd $TMPDIR

# Make a directory save analysis script outputs to
mkdir outputs

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

PYTHON_SCRIPT="pyVertexModel/main.py"

# Run analysis script using Python in activated virtual environment passing in path to
# directory containing input data and path to directory to write outputs to
echo "Running analysis script..."
max_jobs=3
for num_parameter in $(seq 1 $max_jobs); do
    python $PROJECT_DIR/$PYTHON_SCRIPT "$num_parameter" outputs &
    sleep 10s
done
echo "...done."

# Copy script outputs back to scratch space under a job ID specific subdirectory
echo "Copying analysis outputs to scratch space..."
rsync -a outputs/ $PROJECT_DIR/outputs_$JOB_ID/
echo "...done"