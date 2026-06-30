#!/bin/bash
#SBATCH --job-name=srm
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/srm_%j.out
#SBATCH --error=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/srm_%j.err

module load python/anaconda-2023.09
source activate /project/ycleong/users/dsiSummerLab26/env/naturalistic

echo "[INFO] Running srm_only.py"
python /project/ycleong/users/dsiSummerLab26/t-9sador/srm.py
echo "[INFO] Job finished"