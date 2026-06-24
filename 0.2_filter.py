import os
import sys
from nilearn import image

# --- Config ---
data_dir = "/project/ycleong/users/alicial/CS_Learning/preprocessed_data"
output_dir = "/project/ycleong/users/alicial/CS_Learning/preprocessed_data_filtered"
os.makedirs(output_dir, exist_ok=True)

high_pass = 0.008  # Hz
TR = 2.0  # seconds

# --- Subject argument ---
if len(sys.argv) < 2:
    print("Usage: python 0.2_filter.py sub-sXXX")
    sys.exit(1)

sub = sys.argv[1]  # e.g. sub-s102
print(f"Starting filtering for {sub}", flush=True)

# --- Locate subject directory ---
func_root = os.path.join(data_dir, sub, sub)
if not os.path.exists(func_root):
    print(f"{func_root} path does not exist", flush=True)
    sys.exit(1)

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
        print(f"Filtering {input_file}", flush=True)

        output_subdir = os.path.join(output_dir, sub, ses)
        os.makedirs(output_subdir, exist_ok=True)

        output_file = os.path.join(
            output_subdir,
            bold_file.replace("desc-preproc", "desc-filtered")
        )

        # Apply high-pass filter
        filtered_img = image.clean_img(
            input_file,
            detrend=True,
            standardize=False,
            high_pass=high_pass,
            t_r=TR
        )
        filtered_img.to_filename(output_file)
        print(f"Saved filtered file: {output_file}", flush=True)

print(f"Done filtering {sub}", flush=True)