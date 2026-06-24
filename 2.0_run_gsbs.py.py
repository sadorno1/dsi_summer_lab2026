import os
import sys
import numpy as np
import pandas as pd
from statesegmentation import GSBS

# === Command line argument ===
if len(sys.argv) < 2:
    print("Usage: python 2.0_run_gsbs.py <subject_id>")
    sys.exit(1)

subject = sys.argv[1]  # e.g. "sub-s102"
print(f"[INFO] Running GSBS for {subject}")

# === Directories ===
data_dir = "/project/ycleong/users/alicial/CS_Learning/parcellated_data"
output_dir = "/project/ycleong/users/alicial/CS_Learning/gsbs_results"
os.makedirs(output_dir, exist_ok=True)

# === GSBS parameters ===
kmax = 90
statewise_detection = True
finetune = 1
finetune_order = True
dmin = 1

# === Subject directory ===
subj_dir = os.path.join(data_dir, subject)
if not os.path.exists(subj_dir):
    print(f"[ERROR] No directory for {subject}")
    sys.exit(1)

# === Find all ses-wk6 .npy files ===
files = sorted([f for f in os.listdir(subj_dir)
                if "ses-wk6" in f and "recap" in f and f.endswith("_filtered_bold_parcels.npy")])

if not files:
    print(f"[ERROR] No ses-wk6 parcellated files found for {subject}")
    sys.exit(1)

print(f"[INFO] Found {len(files)} parcellated recap runs for {subject}:")
for f in files:
    print(f"   {f}")

results = []

# === Loop through runs ===
for f in files:
    run_name = f.replace(".npy", "")
    file_path = os.path.join(subj_dir, f)
    print(f"[INFO] Loading {run_name}")

    # Load dict of ROI: array(TR × Voxels)
    run_data = np.load(file_path, allow_pickle=True).item()
    if not isinstance(run_data, dict):
        print(f"[WARN] Skipping {run_name}: data is not a dictionary")
        continue

    for roi_name, roi_data in run_data.items():
        if roi_data is None or np.isnan(roi_data).any() or np.allclose(roi_data, 0):
            print(f"[WARN] Skipping {roi_name} in {run_name} (invalid data)")
            continue

        # Average across voxels -> mean timecourse
        # mean_ts = np.mean(roi_data, axis=1, keepdims=True)  # (T, 1)
        # print(f"[INFO] {roi_name}: {roi_data.shape} → mean timecourse shape {mean_ts.shape}")

        # Run GSBS
        gsbs = GSBS()
        gsbs.fit(
            kmax=kmax,
            #x=mean_ts,
            x=roi_data,
            statewise_detection=statewise_detection,
            finetune=finetune,
            finetune_order=finetune_order,
            dmin=dmin
        )

        n_states = len(gsbs.boundaries) + 1
        boundaries = gsbs.boundaries

        results.append({
            "subject": subject,
            "run": run_name,
            "roi": roi_name,
            "n_states": n_states,
            "boundaries": boundaries.tolist()
        })

        print(f"[RESULT] {subject} | {run_name} | {roi_name}: {n_states} states")

# === Save per-subject CSV ===
out_csv = os.path.join(output_dir, f"{subject}_gsbs.csv")
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"[SAVED] {out_csv}")
print("[DONE]")