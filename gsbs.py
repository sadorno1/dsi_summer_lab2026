import os
import pickle
import warnings
import numpy as np
from statesegmentation import GSBS

warnings.simplefilter("ignore")

# CONFIG
condition_label = "ses-wk6_recap"   # IMPORTANT, must match label in srm

srm_out_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/SRM_data"
gsbs_out_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/GSBS_Results_v2"
os.makedirs(gsbs_out_dir, exist_ok=True)

gsbs_kmax = 80
# LOAD SRM PICKLES
print("[STAGE 1] LOADING SRM PICKLES", flush=True)
with open(os.path.join(srm_out_dir, f"students_srm_{condition_label}_200_regions.pkl"), "rb") as f:
    students_srm = pickle.load(f)
with open(os.path.join(srm_out_dir, f"experts_srm_{condition_label}_200_regions.pkl"), "rb") as f:
    experts_srm = pickle.load(f)
with open(os.path.join(srm_out_dir, f"all_srm_{condition_label}_200_regions.pkl"), "rb") as f:
    all_srm = pickle.load(f)
print(f"  students_srm: {len(students_srm)} ROIs", flush=True)
print(f"  experts_srm:  {len(experts_srm)} ROIs", flush=True)
print(f"  all_srm:      {len(all_srm)} ROIs", flush=True)

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

# GSBS
def apply_gsbs_nested(srm_dict, roi, condition, gsbs_results, gsbs_models, kmax=gsbs_kmax):
    roi_data_dict = srm_dict.get(roi, {})
    if not roi_data_dict:
        return
    stacked = np.stack([np.asarray(d, dtype=float) for d in roi_data_dict.values()], axis=0)
    mean_timecourse = np.mean(stacked, axis=0)
    gsbs_model = GSBS(x=mean_timecourse, kmax=kmax, statewise_detection=True)
    gsbs_model.fit()
    gsbs_results[roi] = gsbs_model.nstates
    gsbs_models[roi] = gsbs_model
    print(f"  {condition} - ROI {roi}: optimal n events = {gsbs_model.nstates}", flush=True)

print("\n[STAGE 2] RUNNING GSBS", flush=True)

roi_indices = sorted(set(students_srm) & set(experts_srm) & set(all_srm))
roi_indices = [r for r in roi_indices
               if not (r in students_results and r in experts_results and r in all_results)]
print(f"[INFO] {len(roi_indices)} ROIs remaining to process", flush=True)

for roi in roi_indices:
    apply_gsbs_nested(students_srm, roi, "Students", students_results, students_models)
    apply_gsbs_nested(experts_srm,  roi, "Experts",  experts_results,  experts_models)
    apply_gsbs_nested(all_srm,      roi, "Combined", all_results,      all_models)

    if roi % 10 == 0:
        for name, data in [("students_results", students_results),
                           ("experts_results", experts_results),
                           ("all_results", all_results)]:
            with open(os.path.join(gsbs_out_dir, f"{name}_checkpoint.pkl"), "wb") as f:
                pickle.dump(data, f)
        print(f"[CHECKPOINT] Saved through ROI {roi}", flush=True)

# Final save, now also labeled with condition
data_to_save = {
    f"students_results_{condition_label}": students_results,
    f"students_models_{condition_label}": students_models,
    f"experts_results_{condition_label}":  experts_results,
    f"experts_models_{condition_label}":   experts_models,
    f"all_results_{condition_label}":      all_results,
    f"all_models_{condition_label}":       all_models,
}
for name, data in data_to_save.items():
    with open(os.path.join(gsbs_out_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)
    print(f"[SAVED] {name}.pkl", flush=True)

print("[DONE] GSBS complete.", flush=True)