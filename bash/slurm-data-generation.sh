#!/bin/sh

# The interpreter used to execute the script

# "#SBATCH" directives that convey submission options:

#SBATCH --job-name=data-generation
#SBATCH --nodes=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=100:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=standard
#SBATCH --output=slurm-data-generation.txt

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

cd ~/Documents/Git_repo/pytorch-maml-rl/
srun python3 maml_rl/envs/highway/env_feature_encoder/data_generator_highway.py --config configs/maml/highway.yaml
