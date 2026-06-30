import os
import sys
import numpy as np
import nibabel as nib
from nilearn import datasets, image, masking
from nilearn.image import resample_to_img

# === PATHS ===
data_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/preprocessed_data_filtered"
output_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/parcellated_data"
os.makedirs(output_dir, exist_ok=True)

# === SUBJECT ARGUMENT ===
if len(sys.argv) < 2:
    print("Usage: python parcellate.py sub-sXXX")
    sys.exit(1)

subject = sys.argv[1]
print(f"[INFO] Starting parcellation for {subject}", flush=True)

sub_dir = os.path.join(data_dir, subject)
if not os.path.exists(sub_dir):
    raise FileNotFoundError(f"Missing subject dir for {subject}: {sub_dir}")

out_sub_dir = os.path.join(output_dir, subject)
os.makedirs(out_sub_dir, exist_ok=True)

# === LOAD SCHAEFER 200 / 17-NETWORK ATLAS ===
schaefer = datasets.fetch_atlas_schaefer_2018(
    n_rois=200, yeo_networks=17, resolution_mm=2
)
atlas_img = nib.load(schaefer.maps)

roi_names_all = [
    label.decode("utf-8") if isinstance(label, bytes) else label
    for label in schaefer.labels
]
roi_value_to_name = {i: roi_names_all[i] for i in range(1, 201)}
print(f"[INFO] Loaded Schaefer 200/17Networks atlas with 200 parcels", flush=True)

# === RESAMPLE ATLAS ===
first_bold = None
for ses in sorted(os.listdir(sub_dir)):
    if not ses.startswith("ses-wk"):
        continue
    ses_dir = os.path.join(sub_dir, ses)
    candidates = sorted([f for f in os.listdir(ses_dir)
                         if f.endswith("desc-filtered_bold.nii.gz")])
    if candidates:
        first_bold = os.path.join(ses_dir, candidates[0])
        break

if first_bold is None:
    raise FileNotFoundError(f"No filtered BOLD files found for {subject}")

reference_img = image.load_img(first_bold)
atlas_resamp = resample_to_img(atlas_img, reference_img, interpolation="nearest")
print(f"[INFO] Resampled atlas to BOLD grid using {first_bold}", flush=True)

# === PARCELLATE EACH RUN ===
for ses in sorted(os.listdir(sub_dir)):
    if not ses.startswith("ses-wk"):
        continue

    ses_dir = os.path.join(sub_dir, ses)
    bold_files = sorted([
        f for f in os.listdir(ses_dir)
        if f.endswith("desc-filtered_bold.nii.gz")
    ])

    if not bold_files:
        print(f"[WARN] No filtered BOLD files for {subject} {ses}", flush=True)
        continue

    out_ses_dir = os.path.join(out_sub_dir, ses)
    os.makedirs(out_ses_dir, exist_ok=True)

    for bold_file in bold_files:
        bold_path = os.path.join(ses_dir, bold_file)
        print(f"[INFO] Processing {bold_path}", flush=True)

        bold_img = image.load_img(bold_path)

        # Extract voxels per ROI as (TR, n_voxels)
        roi_arrays = {}
        for roi_value, roi_name in roi_value_to_name.items():
            roi_mask = image.math_img(f"img == {roi_value}", img=atlas_resamp)
            roi_arrays[roi_name] = masking.apply_mask(bold_img, roi_mask)

        out_file = os.path.join(
            out_ses_dir,
            bold_file.replace(
                "desc-filtered_bold.nii.gz",
                "schaefer200_17net_parcels.npz"
            )
        )
        np.savez(out_file, **roi_arrays)
        print(
            f"[SAVED] {out_file} | example ROI shape "
            f"({list(roi_arrays.keys())[0]}): {list(roi_arrays.values())[0].shape}",
            flush=True
        )

print(f"[DONE] All runs processed for {subject}", flush=True)