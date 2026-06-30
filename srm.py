import os
import pickle
import warnings
import numpy as np
from brainiak.funcalign.srm import SRM

from load_data import load_condition_data
warnings.simplefilter("ignore")

# CONFIG
data_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/parcellated_data"
srm_out_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/SRM_data"
os.makedirs(srm_out_dir, exist_ok=True)

recap_weeks = ["ses-wk6"]
recap_tasks = ["wk1recap", "wk2recap",]

srm_n_features = 30
srm_n_iter = 20

# LOAD DATA
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

# SRM
def apply_srm_with_ids(data_dict, feature_num, n_iter=srm_n_iter):
    results, models = {}, {}
    for roi, subjects in data_dict.items():
        if not subjects:
            continue
        subject_ids = list(subjects.keys())
        subject_data = [subjects[sid] for sid in subject_ids]

        n_voxels = subject_data[0].shape[0]
        if n_voxels < feature_num:
            print(f"[SKIP] ROI {roi}: only {n_voxels} voxels < {feature_num} features", flush=True)
            continue
        shapes = {d.shape for d in subject_data}
        if len(shapes) > 1:
            print(f"[WARN] ROI {roi}: inconsistent shapes {shapes}", flush=True)
            continue

        srm = SRM(n_iter=n_iter, features=feature_num)
        srm.fit(subject_data)
        transformed = srm.transform(subject_data)

        results[roi] = {sid: t.T for sid, t in zip(subject_ids, transformed)}
        models[roi] = srm
    return results, models

print("\n[STAGE 2] FITTING SRMs", flush=True)
students_srm, students_srm_models = apply_srm_with_ids(zscored_students, srm_n_features)
# experts_srm,  experts_srm_models  = apply_srm_with_ids(zscored_experts,  srm_n_features)
# all_srm,      all_srm_models      = apply_srm_with_ids(zscored_all,      srm_n_features)

# Save
condition_label = "_".join(recap_weeks) + "_" + "_".join(recap_tasks)
# e.g. "ses-wk6_wk1recap_wk2recap_wk3recap_wk4recap_wk5recap"

srm_to_save = {
    f"students_srm_{condition_label}_200_regions.pkl": students_srm,
    f"students_srm_models_{condition_label}_200_regions.pkl": students_srm_models,
    f"experts_srm_{condition_label}_200_regions.pkl": experts_srm,
    f"experts_srm_models_{condition_label}_200_regions.pkl": experts_srm_models,
    f"all_srm_{condition_label}_200_regions.pkl": all_srm,
    f"all_srm_models_{condition_label}_200_regions.pkl": all_srm_models,
}

for name, obj in srm_to_save.items():
    with open(os.path.join(srm_out_dir, name), "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVED] {name}", flush=True)

print("[DONE] SRM complete.", flush=True)