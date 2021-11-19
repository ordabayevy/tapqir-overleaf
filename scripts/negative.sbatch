#!/bin/bash
#SBATCH --job-name=negative
#SBATCH --output=simulations/negative%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-4

lamdas=(0.01 0.05 0.15 0.5 1)

# Path to your executable
python scripts/cosmos_simulations.py \
  --gain 7 --pi 0 --height 3000 --proximity 0.2 \
  --lamda ${lamdas[${SLURM_ARRAY_TASK_ID}]} \
  --cuda \
  --path simulations/negative${lamdas[${SLURM_ARRAY_TASK_ID}]}
