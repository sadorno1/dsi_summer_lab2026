#!/bin/bash
# Submits run_parcellate.sh as a separate SLURM job for every subject.

SUBJECTS=(
    sub-s102 sub-s103 sub-s105 sub-s106 sub-s107
    sub-s108 sub-s110 sub-s111 sub-s112 sub-s113
    sub-s114 sub-s116 sub-s118 sub-s120 sub-s121
    sub-s122 sub-s125 sub-s126 sub-s127 sub-s129
    sub-s201 sub-s213 sub-s214 sub-s215 sub-s216
)

echo "[INFO] Submitting parcellate jobs for ${#SUBJECTS[@]} subjects"

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "[INFO] Submitting $SUBJECT"
    sbatch run_parcellate.sh "$SUBJECT"
    sleep 1   # small delay to avoid hammering the scheduler
done

echo "[INFO] All jobs submitted. Check with: squeue -u t-9sador"