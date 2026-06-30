#!/bin/bash
#SBATCH --job-name=plot_events
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/plot_%j.out
#SBATCH --error=/project/ycleong/users/dsiSummerLab26/t-9sador/logs/plot_%j.err

module load python/anaconda-2023.09
source activate naturalistic

# /project/ycleong/users/dsiSummerLab26/env/naturalistic/bin/python plot_brain.py
/project/ycleong/users/dsiSummerLab26/env/naturalistic/bin/python analyze_n_events.py
# /project/ycleong/users/dsiSummerLab26/env/naturalistic/bin/python plot_correlation_between_Ks_.py
# /project/ycleong/users/dsiSummerLab26/env/naturalistic/bin/python plot_spatial_correlation.py
