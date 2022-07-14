#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/main_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G

#Variables
myDir="/home/yoogo/projects/def-seanwood/yoogo/mywork/keyword_spotting"
archiveFile="$myDir/speech_commands_v0.02.tar.gz"
root=${SLURM_TMPDIR}
archiveDir="$root/SpeechCommands/speech_commands_v0.02"

mkdir -p $archiveDir
tar -xf $archiveFile --directory $archiveDir

source ~/venv/bin/activate
python main.py --model MFCC --num-epochs 100 --exp_name test
python main.py --model mel --num-epochs 100 --exp_name test
python main.py --model M5 --num-epochs 100 --exp_name test
deactivate