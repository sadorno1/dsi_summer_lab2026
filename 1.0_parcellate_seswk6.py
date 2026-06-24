import os
import numpy as np
import nibabel as nib
from nilearn import image, masking

# === PATHS ===
data_dir = "/project/ycleong/users/alicial/CS_Learning/preprocessed_data_filtered"
atlas_dir = "/project/ycleong/users/alicial/resources/atlases"
output_dir = "/project/ycleong/users/alicial/CS_Learning/parcellated_data"
os.makedirs(output_dir, exist_ok=True)

# === LOAD ROI MASKS ===
rois = [f for f in os.listdir(atlas_dir) if f.startswith("ROI_") and f.endswith("_2mm.nii.gz")]
roi_dict = {os.path.splitext(f)[0].replace("ROI_", ""): os.path.join(atlas_dir, f) for f in rois}
print(f"[INFO] Found {len(roi_dict)} ROI masks: {list(roi_dict.keys())}")

# === PARAMETERS ===
subject = os.environ.get("SUBJECT")  # passed in via bash
if subject is None:
    raise RuntimeError("Please pass SUBJECT environment variable.")

sub_dir = os.path.join(data_dir, subject, "ses-wk6")
if not os.path.exists(sub_dir):
    raise FileNotFoundError(f"Missing ses-wk6 for {subject}")

out_sub_dir = os.path.join(output_dir, subject)
os.makedirs(out_sub_dir, exist_ok=True)

# === FIND FILTERED FILES ===
bold_files = sorted([
    f for f in os.listdir(sub_dir)
    if f.endswith("desc-filtered_bold.nii.gz")
])

if not bold_files:
    raise FileNotFoundError(f"No filtered BOLD files for {subject} in ses-wk6")

print(f"[INFO] Found {len(bold_files)} runs for {subject}")

# === PARCELLATION ===
for bold_file in bold_files:
    bold_path = os.path.join(sub_dir, bold_file)
    print(f"[INFO] Processing {bold_path}")

    img = image.load_img(bold_path)
    roi_tc = {}  # store mean time courses

    for roi_name, roi_path in roi_dict.items():
        roi_img = image.load_img(roi_path)

        # Resample to match fMRI resolution if needed
        roi_resamp = image.resample_to_img(roi_img, img, interpolation="nearest")

        # Extract ROI mask and apply
        roi_mask = masking.apply_mask(img, roi_resamp)

        # Mean signal across voxels, then z-score across time
        mean_tc = np.mean(roi_mask, axis=1)
        zscore_tc = (mean_tc - np.mean(mean_tc)) / np.std(mean_tc)

        roi_tc[roi_name] = zscore_tc

    # Save each ROI’s z-scored timecourse
    out_file = os.path.join(out_sub_dir, bold_file.replace(".nii.gz", "_parcels.npy"))
    np.save(out_file, roi_tc)
    print(f"[SAVED] {out_file}")

print("[DONE] All runs processed.")