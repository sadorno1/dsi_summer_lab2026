import os
import sys
from nilearn import image, masking

# --- Config ---
data_dir = "/project/ycleong/users/alicial/CS_Learning/preprocessed_data"
output_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning_GSR/preprocessed_data_filtered_gsr"
os.makedirs(output_dir, exist_ok=True)

high_pass = 0.008  # Hz
TR = 2.0  # seconds

# --- Subject argument ---
if len(sys.argv) < 2:
    print("Usage: python filter_gsr.py sub-sXXX")
    sys.exit(1)

sub = sys.argv[1]  # e.g. sub-s102
print(f"Starting GSR filtering for {sub}", flush=True)

# --- Locate subject directory ---
func_root = os.path.join(data_dir, sub, sub)
if not os.path.exists(func_root):
    print(f"{func_root} path does not exist", flush=True)
    sys.exit(1)


def compute_global_signal(input_file, mask_file=None):
    """Compute mean signal across all in-brain voxels at each timepoint.

    Returns shape (TR, 1), ready to pass to clean_img(confounds=...).
    """
    if mask_file is not None and os.path.exists(mask_file):
        data = masking.apply_mask(input_file, mask_file)  # (TR, n_voxels)
    else:
        from nilearn.masking import compute_epi_mask
        mask_img = compute_epi_mask(input_file)
        data = masking.apply_mask(input_file, mask_img)
    global_signal = data.mean(axis=1)  # (TR,)
    return global_signal.reshape(-1, 1)


# --- Iterate through sessions ---
for ses in sorted(os.listdir(func_root)):
    if not ses.startswith("ses-wk"):
        continue

    func_dir = os.path.join(func_root, ses, "func")
    if not os.path.exists(func_dir):
        continue

    bold_files = [f for f in os.listdir(func_dir) if f.endswith("desc-preproc_bold.nii.gz")]
    if not bold_files:
        print(f"No functional files found for {sub} {ses}", flush=True)
        continue

    for bold_file in bold_files:
        input_file = os.path.join(func_dir, bold_file)
        print(f"Filtering (GSR) {input_file}", flush=True)

        # Look for an fMRIPrep brain mask sitting next to the bold file
        mask_candidate = input_file.replace("desc-preproc_bold.nii.gz", "desc-brain_mask.nii.gz")
        mask_file = mask_candidate if os.path.exists(mask_candidate) else None
        if mask_file:
            print(f"  Using mask: {mask_file}", flush=True)
        else:
            print(f"  No mask found at {mask_candidate}, auto-computing EPI mask", flush=True)

        global_signal = compute_global_signal(input_file, mask_file)

        output_subdir = os.path.join(output_dir, sub, ses)
        os.makedirs(output_subdir, exist_ok=True)

        output_file = os.path.join(
            output_subdir,
            bold_file.replace("desc-preproc", "desc-filteredGSR")
        )

        filtered_img = image.clean_img(
            input_file,
            detrend=True,
            standardize=False,
            high_pass=high_pass,
            t_r=TR,
            confounds=global_signal
        )
        filtered_img.to_filename(output_file)
        print(f"Saved GSR-filtered file: {output_file}", flush=True)

print(f"Done GSR filtering {sub}", flush=True)