#!/bin/bash
#SBATCH --job-name=kinetic
#SBATCH --output=simulations/kinetic%a_%A.out
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --array=0-11

kon=(0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.03 0.03 0.03 0.03)
lamda=(0.01 0.15 0.5 1 0.01 0.15 0.5 1 0.01 0.15 0.5 1 0.01 0.15 0.5 1)

# Path to your executable
python scripts/kinetic_simulations.py \
  -N 100 -F 1000 -bs 16 --cuda --dtype double \
  --kon ${kon[${SLURM_ARRAY_TASK_ID}]} \
  --lamda ${lamda[${SLURM_ARRAY_TASK_ID}]} \
  --path simulations/kon${kon[${SLURM_ARRAY_TASK_ID}]}lamda${lamda[${SLURM_ARRAY_TASK_ID}]}
