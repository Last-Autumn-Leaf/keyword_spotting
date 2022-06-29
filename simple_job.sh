#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=def-seanwood
#SBATCH --output=./logs/test_%j.out
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G

myDir="/home/yoogo/projects/def-seanwood/yoogo/mywork/keyword_spotting"
#python_file="$myDir/test.py"
archiveFile="$myDir/speech_commands_v0.02.tar.gz"
#--------- unzipping dataset --------
cd $SLURM_TMPDIR

cp archiveFile $SLURM_TMPDIR
#mkdir SpeechCommands
#cd SpeechCommands

#tar -xf $archiveFile

#--------- unzipping dataset --------

cd $myDir
source ~/venv/bin/activate
python test.py $SLURM_TMPDIR

deactivate

