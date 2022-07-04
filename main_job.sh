#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/main_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G

myDir="/home/yoogo/projects/def-seanwood/yoogo/mywork/keyword_spotting"
archiveFile="$myDir/speech_commands_v0.02.tar.gz"

cp $archiveFile $SLURM_TMPDIR

source ~/venv/bin/activate
python main.py --model PDM mel --save_checkpoint 'PDM' --num-epochs 10 --lr 0.00001 --pdm_factor 48

deactivate
