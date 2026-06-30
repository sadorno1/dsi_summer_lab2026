import os
import glob
import numpy as np
from nilearn import datasets
from scipy.stats import zscore

# subjects

ALL_SUBJECTS = [
    "sub-s102", "sub-s103", "sub-s105", "sub-s106", "sub-s107",
    "sub-s108", "sub-s110", "sub-s111", "sub-s112", "sub-s113",
    "sub-s114", "sub-s116", "sub-s118", "sub-s120", "sub-s121",
    "sub-s122", "sub-s125", "sub-s126", "sub-s127", "sub-s129",
    "sub-s201", "sub-s213", "sub-s214", "sub-s215", "sub-s216"
]

def get_subject_ids(group='combined', exclude=None):
    if exclude is None:
        exclude = []

    ids = [s for s in ALL_SUBJECTS if s not in exclude]

    if group == 'combined':
        return ids
    elif group == 'students':
        return [s for s in ids if s.startswith("sub-s1")]
    elif group == 'experts':
        return [s for s in ids if s.startswith("sub-s2")]
    else:
        raise ValueError("group must be 'combined', 'students', or 'experts'")


# atlas

def get_roi_name_to_value():
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=2)
    labels = [l.decode("utf-8") if isinstance(l, bytes) else l for l in schaefer.labels]
    return {labels[i]: i for i in range(1, 201)}

ROI_NAME_TO_VALUE = get_roi_name_to_value()
ROI_INDICES = list(range(1, 201))


# loader

def load_condition_data(group, base_dir, weeks, task_identifiers, exact_match=False, exclude=None, verbose=True):
    """
    Finds and grabs matching raw files for ALL valid subjects across groups, 
    identifies and drops dead voxels based on that pooled data, 
    individually z-scores each timecourse, and filters down to the requested group.
    """
    # 1. Gather files for the entire dataset pool to check masks properly
    all_subject_ids = get_subject_ids(group='combined', exclude=exclude)
    roi_name_to_value = ROI_NAME_TO_VALUE
    raw_pool = {roi: {} for roi in ROI_INDICES}

    for subj in all_subject_ids:
        subj_path = os.path.join(base_dir, subj)
        if not os.path.isdir(subj_path):
            continue

        matched_files = []
        for week in weeks:
            week_path = os.path.join(subj_path, week)
            if not os.path.isdir(week_path):
                continue
            all_npz = glob.glob(os.path.join(week_path, "*.npz"))
            for fpath in all_npz:
                fname = os.path.basename(fpath)
                match_condition = (any(f"task-{tid}" in fname for tid in task_identifiers) if exact_match 
                                   else any(tid in fname for tid in task_identifiers))
                if match_condition:
                    matched_files.append(fpath)

        if not matched_files:
            continue

        roi_arrays = {roi: [] for roi in ROI_INDICES}
        for fpath in sorted(matched_files):
            data = np.load(fpath, allow_pickle=True)
            for roi_name, arr in data.items():
                if roi_name in roi_name_to_value:
                    roi_arrays[roi_name_to_value[roi_name]].append(arr)

        for roi_idx, list_of_arrays in roi_arrays.items():
            if list_of_arrays:
                raw_pool[roi_idx][subj] = np.concatenate(list_of_arrays, axis=0)

    # 2. Automatically remove dead voxels, z-score, and isolate the group requested
    cleaned_group_data = {roi: {} for roi in ROI_INDICES}
    target_group_ids = get_subject_ids(group=group, exclude=exclude)

    for roi, subjects in raw_pool.items():
        if not subjects:
            continue
            
        # Drop zero-variance voxels relative to the active task pool
        stacked = np.stack(list(subjects.values()), axis=0)
        valid_voxels_mask = np.all(np.std(stacked, axis=1) != 0, axis=0)

        # Transpose and isolate requested subjects
        for subject_id, subject_data in subjects.items():
            if subject_id in target_group_ids:
                zscored = zscore(subject_data[:, valid_voxels_mask], axis=0)
                
                if np.isnan(zscored).any() and verbose:
                    print(f"[WARN] NaNs encountered in ROI {roi} for subject {subject_id}")
                    
                cleaned_group_data[roi][subject_id] = zscored.T

  
        
    if verbose:
        print(f"[INFO] Successfully loaded, cleaned, and z-scored '{group}' data.")
        if 1 in cleaned_group_data and cleaned_group_data[1]:
            example_shape = next(iter(cleaned_group_data[1].values())).shape
            print(f"[INFO] Target output shape format: {example_shape} (n_voxels, TR)")
        
    return cleaned_group_data

       