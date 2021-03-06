#!/bin/sh

# The interpreter used to execute the script

# "#SBATCH" directives that convey submission options:

#SBATCH --job-name=maml-evaluate
#SBATCH --nodes=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=100:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=standard
#SBATCH --output=slurm-eval-output.txt

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

cd ~/Documents/Git_repo/pytorch-maml-rl/
srun python3 test.py --config configs/maml/highway.yaml --output-folder maml-highway/change_reward/08212020_1 --policy maml-highway/change_reward/08212020_1/policy99.th --num-batches 10000
