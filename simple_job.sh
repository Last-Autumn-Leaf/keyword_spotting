#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=def-seanwood
#SBATCH --array=1-10
#SBATCH --output=./logs/test_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
#Variables
myDir="/home/yoogo/projects/def-seanwood/yoogo/mywork/keyword_spotting"
archiveFile="$myDir/speech_commands_v0.02.tar.gz"
root=${SLURM_TMPDIR}
archiveDir="$root/SpeechCommands/speech_commands_v0.02"

mkdir -p $archiveDir
tar -xf $archiveFile --directory $archiveDir

source ~/venv/bin/activate

#---------------
python test.py $SLURM_ARRAY_TASK_ID