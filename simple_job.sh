#!/bin/bash
#SBATCH --time=00:2:00
#SBATCH --account=def-seanwood
#SBATCH --output=./output/slurm_%j.out
#SBTACH --mem=125G
python test.py
sleep 30
