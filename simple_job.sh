#!/bin/bash
#SBATCH --time=00:2:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/slurm_%j.out
#SBTACH --cpus-per-task=3
#SBTACH --mem=12G
source ~/venv/bin/activate
python test.py
deactivate
sleep 30
