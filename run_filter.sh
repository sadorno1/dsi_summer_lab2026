#!/bin/bash
#SBATCH --job-name=filter_s102
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/filter_%j.out
#SBATCH --error=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/filter_%j.err


SUBJECT=$1

if [ -z "$SUBJECT" ]; then
    echo "Usage: sbatch run_filter.sh sub-sXXX"
    exit 1
fi

module load python/anaconda-2023.09
source activate naturalistic
/project/ycleong/users/dsiSummerLab26/env/naturalistic/bin/python /project/ycleong/users/dsiSummerLab26/t-9sador/global_signal_regression/filter_grs.py "$SUBJECT"
