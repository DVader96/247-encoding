#!/bin/bash
#SBATCH --time 1:10:00
#SBATCH --mem 30GB
#SBATCH --cpus-per-task=2


module load anaconda
source activate torch_env

echo 'Start time:' `date`
echo "$@"
python brain_color_prepro.py "$@"
echo 'End time:' `date`
