#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/test_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
source ~/venv/bin/activate
python test.py
deactivate
sleep 30
