#!/bin/bash
#SBATCH --time=00:2:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/slurm_%j.out
#SBTACH --mem=16G
python test.py
sleep 30