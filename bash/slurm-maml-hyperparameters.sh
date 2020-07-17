#!/bin/sh

# The interpreter used to execute the script

# "#SBATCH" directives that convey submission options:

#SBATCH --job-name=maml-hyperparameters
#SBATCH --nodes=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=100:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=standard
#SBATCH --output=slurm-output.txt

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

cd ~/Documents/Git_repo/pytorch-maml-rl/
srun python3 train.py --config configs/maml/highway.yaml --output-folder maml-highway/batch100x40 --seed 1 --num-workers 8
srun python3 train.py --config configs/maml/highway_lr01.yaml --output-folder maml-highway/lr01 --seed 1 --num-workers 8
srun python3 train.py --config configs/maml/highway_lr001.yaml --output-folder maml-highway/lr001 --seed 1 --num-workers 8
srun python3 train.py --config configs/maml/highway_first_order_app.yaml --output-folder maml-highway/first_order_app --seed 1 --num-workers 8
wait