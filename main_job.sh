#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/slurm_%j.out
#SBTACH --cpus-per-task=3
#SBTACH --mem=12G
source ~/venv/bin/activate
python main.py --model mel --save_checkpoint 'mel' --num-epochs 10