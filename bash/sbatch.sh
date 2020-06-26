#!/bin/sh

# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=test-maml
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=10:00
#SBATCH --account=hpeng1
#SBATCH --partition=standard

cd ~/Documents/Git_repo/pytorch-maml-rl/
python test.py --config configs/maml/highway-eval.yaml --output-folder maml-highway-eval --policy maml-highway/04222020/policy.th --seed 1 --num-workers 8
