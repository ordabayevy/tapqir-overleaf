#!/bin/bash
#SBATCH --job-name=datasetB
#SBATCH --output=experimental/datasetB.out
#SBATCH --gres=gpu:1   # Request 1 GPU

python scripts/ppc.py
