#!/bin/bash
#SBATCH --job-name=parcellate
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/parcellate_%x_%j.out
#SBATCH --error=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/parcellate_%x_%j.err

# Usage: sbatch run_parcellate.sh sub-s102

SUBJECT=$1

if [ -z "$SUBJECT" ]; then
    echo "Usage: sbatch run_parcellate.sh sub-sXXX"
    exit 1
fi

module load python/anaconda-2023.09
source activate naturalistic

echo "[INFO] Running parcellate.py for $SUBJECT"
/project/ycleong/users/dsiSummerLab26/env/naturalistic/bin/python /project/ycleong/users/dsiSummerLab26/t-9sador/parcellate.py "$SUBJECT"

echo "[INFO] Job finished for $SUBJECT"

