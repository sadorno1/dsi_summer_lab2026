import os
import pickle
import warnings
import numpy as np
from statesegmentation import GSBS
from load_data import load_condition_data

warnings.simplefilter("ignore")

# CONFIG
data_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/parcellated_data"
gsbs_out_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/GSBS_Results_z_scored"
os.makedirs(gsbs_out_dir, exist_ok=True)

recap_weeks = ["ses-wk6"]
recap_tasks = ["wk1recap", "wk2recap", "wk3recap", "wk4recap", "wk5recap"]
GSBS_KMAX = 80

condition_label = "_".join(recap_weeks) + "_recap"

# LOAD
print("[STAGE 1] LOADING DATA", flush=True)
zscored_students = load_condition_data(
    group='students', base_dir=data_dir,
    weeks=recap_weeks, task_identifiers=recap_tasks, verbose=True
)
zscored_experts = load_condition_data(
    group='experts', base_dir=data_dir,
    weeks=recap_weeks, task_identifiers=recap_tasks, verbose=True
)
zscored_all = load_condition_data(
    group='combined', base_dir=data_dir,
    weeks=recap_weeks, task_identifiers=recap_tasks, verbose=True
)

# === RESUME FROM CHECKPOINT IF AVAILABLE ===
students_results, students_models = {}, {}
experts_results,  experts_models  = {}, {}
all_results,      all_models      = {}, {}

for name, results_dict in [("students_results", students_results),
                            ("experts_results", experts_results),
                            ("all_results", all_results)]:
    ckpt_path = os.path.join(gsbs_out_dir, f"{name}_checkpoint.pkl")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            results_dict.update(pickle.load(f))
        print(f"[RESUME] Loaded {len(results_dict)} ROIs from {name}_checkpoint.pkl", flush=True)
    else:
        print(f"[RESUME] No checkpoint found for {name}, starting fresh", flush=True)

# GSBS on raw z-scored data
def apply_gsbs_raw(data_dict, roi, condition, gsbs_results, gsbs_models, kmax=GSBS_KMAX):
    """Run GSBS on across-subject mean of raw z-scored timecourses for one ROI.
    Input shape per subject: (n_voxels, TR). Mean is taken across subjects,
    then transposed to (TR, n_voxels) for GSBS.
    """
    roi_data_dict = data_dict.get(roi, {})
    if not roi_data_dict:
        return

    # stack: (n_subjects, n_voxels, TR)
    stacked = np.stack([np.asarray(d, dtype=float) for d in roi_data_dict.values()], axis=0)
    # mean across subjects: (n_voxels, TR) -> transpose to (TR, n_voxels)
    mean_timecourse = np.mean(stacked, axis=0).T

    gsbs_model = GSBS(x=mean_timecourse, kmax=kmax, statewise_detection=True)
    gsbs_model.fit()
    gsbs_results[roi] = gsbs_model.nstates
    gsbs_models[roi] = gsbs_model
    print(f"  {condition} - ROI {roi}: optimal n events = {gsbs_model.nstates}", flush=True)

print("\n[STAGE 2] RUNNING GSBS ON RAW Z-SCORED DATA", flush=True)

roi_indices = sorted(set(zscored_students) & set(zscored_experts) & set(zscored_all))
roi_indices = [r for r in roi_indices
               if not (r in students_results and r in experts_results and r in all_results)]
print(f"[INFO] {len(roi_indices)} ROIs remaining to process", flush=True)

for roi in roi_indices:
    apply_gsbs_raw(zscored_students, roi, "Students", students_results, students_models)
    apply_gsbs_raw(zscored_experts,  roi, "Experts",  experts_results,  experts_models)
    apply_gsbs_raw(zscored_all,      roi, "Combined", all_results,      all_models)

    if roi % 10 == 0:
        for name, data in [("students_results", students_results),
                            ("experts_results",  experts_results),
                            ("all_results",      all_results)]:
            with open(os.path.join(gsbs_out_dir, f"{name}_checkpoint.pkl"), "wb") as f:
                pickle.dump(data, f)
        print(f"[CHECKPOINT] Saved through ROI {roi}", flush=True)

# Final save
data_to_save = {
    f"students_results_{condition_label}": students_results,
    f"students_models_{condition_label}":  students_models,
    f"experts_results_{condition_label}":  experts_results,
    f"experts_models_{condition_label}":   experts_models,
    f"all_results_{condition_label}":      all_results,
    f"all_models_{condition_label}":       all_models,
}
for name, data in data_to_save.items():
    with open(os.path.join(gsbs_out_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVED] {name}.pkl", flush=True)

print("[DONE] GSBS on raw data complete.", flush=True)