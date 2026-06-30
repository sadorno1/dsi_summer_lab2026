#!/bin/bash
#SBATCH --job-name=gsbs_r
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/gsbs_%j.out
#SBATCH --error=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/gsbs_%j.err

module load python/anaconda-2023.09
source activate /project/ycleong/users/dsiSummerLab26/env/naturalistic

echo "[INFO] Running gsbs_only.py"
/project/ycleong/users/dsiSummerLab26/env/naturalistic/bin/python  /project/ycleong/users/dsiSummerLab26/t-9sador/gsbs_raw.py
echo "[INFO] Job finished"

