#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
source ~/venv/bin/activate
python main.py --model PDM --save_checkpoint 'PDM' --num-epochs 100
