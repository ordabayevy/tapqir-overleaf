#!/bin/bash
#SBATCH --job-name=seed
#SBATCH --output=simulations/seed%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-16

# Path to your executable
python scripts/cosmos_simulations.py \
  --seed ${SLURM_ARRAY_TASK_ID} \
  --cuda \
  --path simulations/seed${SLURM_ARRAY_TASK_ID}
