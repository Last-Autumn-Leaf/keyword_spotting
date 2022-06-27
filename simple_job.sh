#!/bin/bash
#SBATCH --time=00:2:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
source ~/venv/bin/activate
python ./dataset/subsetSC.py
deactivate
sleep 30
