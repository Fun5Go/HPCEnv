#!/bin/bash

#!/bin/bash

# Name of the job
#SBATCH -J RL_Training

# -------------- Set job requirements -------------- #
# Duration for which the nodes remain allocated D-HH:MM:SS
#SBATCH -t 01:00:00

# Set the partition you want to submit
#SBATCH -p gpu_h100

# Number of nodes
#SBATCH -N 1

# Number of tasks
##SBATCH -n 16

# Tasks/processes per node (1 CPU per task/process)
##SBATCH --tasks-per-node 16

# Different number of processors per task
##SBATCH --ntasks=8
#SBATCH --cpus-per-task 16

# Number of GPUS for the job
#SBATCH --gpus=1

# Number of GPUS per node
##SBATCH --gpus-per-node=4

# -------------- Loading modules -------------- #
module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# -------------- Code -------------- #
# Load virtual environment
source $HOME/.venv/RL/bin/activate

# Run training
bash main.sh
