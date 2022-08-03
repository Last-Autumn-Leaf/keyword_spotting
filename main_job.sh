#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/main_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --array=0-9
#Variables
myDir="/home/yoogo/projects/def-seanwood/yoogo/mywork/keyword_spotting"
archiveFile="$myDir/speech_commands_v0.02.tar.gz"
root=${SLURM_TMPDIR}
archiveDir="$root/SpeechCommands/speech_commands_v0.02"

mkdir -p $archiveDir
tar -xf $archiveFile --directory $archiveDir

source ~/venv/bin/activate
python run_array_PDM.py $SLURM_ARRAY_TASK_ID
