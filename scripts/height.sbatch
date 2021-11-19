#!/bin/bash
#SBATCH --job-name=height
#SBATCH --output=simulations/height%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-7

heights=(300 500 600 750 1000 1500 2000 3000)

# Path to your executable
python scripts/cosmos_simulations.py \
  --gain 7 --pi 0.15 --lamda 0.15 --proximity 0.2 \
  --height ${heights[${SLURM_ARRAY_TASK_ID}]} \
  --cuda \
  --path simulations/height${heights[${SLURM_ARRAY_TASK_ID}]}
