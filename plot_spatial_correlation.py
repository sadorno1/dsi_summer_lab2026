import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_condition_data

# === CONFIGURATION ===
PARCELLATED_ROOT = "/project/ycleong/users/dsiSummerLab26/CS_Learning/parcellated_data"
SRM_DIR = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/SRM_data"
OUTPUT_DIR = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECAP_WEEKS = ["ses-wk6"]
RECAP_TASKS = ["wk1recap", "wk2recap", "wk3recap", "wk4recap", "wk5recap"]

DATA_SOURCE = "zscored"   # "zscored" or "srm"
target_roi = 80

# === 1. LOAD DATA ===
if DATA_SOURCE == "zscored":
    print("[INFO] Loading z-scored voxel data...")
    students_data = load_condition_data(
        group='students', base_dir=PARCELLATED_ROOT,
        weeks=RECAP_WEEKS, task_identifiers=RECAP_TASKS, verbose=False,
    )
    experts_data = load_condition_data(
        group='experts', base_dir=PARCELLATED_ROOT,
        weeks=RECAP_WEEKS, task_identifiers=RECAP_TASKS, verbose=False,

    )
    # shape per subject: (n_voxels, TR),  need rowvar=False for TR×TR corr
    transpose_for_corr = True

elif DATA_SOURCE == "srm":
    print("[INFO] Loading SRM-projected data...")
    with open(os.path.join(SRM_DIR, "students_srm_200_regions.pkl"), "rb") as f:
        students_data = pickle.load(f)
    with open(os.path.join(SRM_DIR, "experts_srm_200_regions.pkl"), "rb") as f:
        experts_data = pickle.load(f)
    # shape per subject: (TR, features),  rows are TRs, default corrcoef works
    transpose_for_corr = False

else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")

# === 2. AGGREGATE ACROSS SUBJECTS ===
if target_roi not in students_data or target_roi not in experts_data:
    raise KeyError(f"ROI {target_roi} not found in one of the data dicts (may have been skipped during SRM)")

student_mats = list(students_data[target_roi].values())
expert_mats  = list(experts_data[target_roi].values())

student_group_mean = np.mean(np.stack(student_mats, axis=0), axis=0)
expert_group_mean  = np.mean(np.stack(expert_mats,  axis=0), axis=0)

# === 3. COMPUTE TIMEPOINT CORRELATION (TR x TR) ===
if transpose_for_corr:
    student_corr = np.corrcoef(student_group_mean, rowvar=False)
    expert_corr  = np.corrcoef(expert_group_mean,  rowvar=False)
else:
    student_corr = np.corrcoef(student_group_mean)
    expert_corr  = np.corrcoef(expert_group_mean)

# === 4. PLOT ===
print("[INFO] Generating correlation plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

im0 = axes[0].imshow(student_corr, cmap='viridis', origin='lower', vmin=-0.8, vmax=0.8)
axes[0].set_title(f'Students Spatial Pattern Correlation (ROI {target_roi}, {DATA_SOURCE}, {RECAP_TASKS})')
axes[0].set_xlabel('Timepoint (TR)')
axes[0].set_ylabel('Timepoint (TR)')
fig.colorbar(im0, ax=axes[0], shrink=0.7)

im1 = axes[1].imshow(expert_corr, cmap='viridis', origin='lower', vmin=-0.8, vmax=0.8)
axes[1].set_title(f'Experts Spatial Pattern Correlation (ROI {target_roi}, {DATA_SOURCE}, {RECAP_TASKS})')
axes[1].set_xlabel('Timepoint (TR)')
axes[1].set_ylabel('Timepoint (TR)')
fig.colorbar(im1, ax=axes[1], shrink=0.7)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, f"spatial_pattern_correlation_roi_{target_roi}_{DATA_SOURCE}_{RECAP_TASKS}_demean.png")
plt.savefig(out_path, dpi=300)
print(f"[SAVED] Figure saved to: {out_path}")