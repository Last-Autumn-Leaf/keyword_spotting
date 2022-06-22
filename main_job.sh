#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
source ~/venv/bin/activate
python main.py --model mel --save_checkpoint 'mel' --num-epochs 10