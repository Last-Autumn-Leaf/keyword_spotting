#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/slurm_%j.out
#SBTACH --mem=16G
python main.py --model mel --save_checkpoint 'mel' --num-epochs 10