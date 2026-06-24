# /project/ycleong/users/alicial/CS_Learning/scripts/create_8ROI_HO_thr20_fixed.py
import os
import numpy as np
import nibabel as nib
from nilearn import datasets, plotting
from nilearn import image

ATLAS_DIR = "/project/ycleong/users/alicial/resources/atlases"
OUT_DIR = ATLAS_DIR  # save masks here
PLOT_DIR = os.path.join(ATLAS_DIR, "roi_plots_HOspace")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("[INFO] Loading Harvard–Oxford probabilistic 2mm atlases (cached/offline if already present)...")
cort = datasets.fetch_atlas_harvard_oxford("cort-prob-2mm", symmetric_split=False, data_dir=ATLAS_DIR)
sub  = datasets.fetch_atlas_harvard_oxford("sub-prob-2mm",  symmetric_split=False, data_dir=ATLAS_DIR)
print(f"[INFO] Cortical labels: {len(cort['labels'])}, volumes: {np.asanyarray(cort['maps'].get_fdata()).shape[-1]}")
print(f"[INFO] Subcortical labels: {len(sub['labels'])}, volumes: {np.asanyarray(sub['maps'].get_fdata()).shape[-1]}")

def as_img(obj):
    if isinstance(obj, nib.spatialimages.SpatialImage):
        return obj
    if isinstance(obj, list):
        return nib.load(obj[0])
    if isinstance(obj, str):
        return nib.load(obj)
    raise TypeError(f"Unsupported maps type: {type(obj)}")

cort_img = as_img(cort["maps"])
sub_img  = as_img(sub["maps"])
cort_data = cort_img.get_fdata()
sub_data  = sub_img.get_fdata()

# Build robust label ↔ volume index maps (handles "Background" at labels[0])
def build_label_index(labels, n_vols):
    # why: HO labels often include an extra "Background" entry
    offset = 1 if len(labels) == n_vols + 1 else 0
    mapping = {}
    for vol_idx in range(n_vols):
        label_idx = vol_idx + offset
        name = labels[label_idx]
        mapping[name.lower()] = vol_idx
    return mapping

cort_map = build_label_index(cort["labels"], cort_data.shape[3])
sub_map  = build_label_index(sub["labels"],  sub_data.shape[3])

# Target 8 bilateral ROIs (HO exact names)
ROI_DEFS = {
    "AngularGyrus": ["Angular Gyrus"],
    "Precuneus":    ["Precuneous Cortex"],
    "ACC":          ["Cingulate Gyrus, anterior division"],
    "Hippocampus":  ["Left Hippocampus", "Right Hippocampus"],
    "pSTG":         ["Superior Temporal Gyrus, posterior division"],
    "Visual":       ["Intracalcarine Cortex"],
    "Auditory":     ["Heschl's Gyrus (includes H1 and H2)"],
    "Amygdala":     ["Left Amygdala", "Right Amygdala"],
}

def get_mask_from_names(names, atlas_kind):
    # why: search both atlases; names may be cortical or subcortical
    mask_accum = None
    for name in names:
        lname = name.lower()
        if lname in cort_map:
            vol = cort_data[..., cort_map[lname]]
            pass_img = cort_img
        elif lname in sub_map:
            vol = sub_data[..., sub_map[lname]]
            pass_img = sub_img
        else:
            print(f"[WARN] Label not found: {name}")
            continue
        bin_mask = (vol > 20).astype(np.uint8)  # >20% threshold
        mask_accum = bin_mask if mask_accum is None else np.maximum(mask_accum, bin_mask)
    if mask_accum is None:
        return None, None
    return nib.Nifti1Image(mask_accum, pass_img.affine, pass_img.header), pass_img

# Create masks and save + quick QA plot in HO space (no template mismatch)
for roi, names in ROI_DEFS.items():
    print(f"[INFO] ROI: {roi}  ← {names}")
    mask_img, ref_img = get_mask_from_names(names, atlas_kind="auto")
    if mask_img is None:
        print(f"[ERROR] Could not build mask for ROI: {roi}")
        continue
    out_mask = os.path.join(OUT_DIR, f"ROI_{roi}_thr20_2mm.nii.gz")
    nib.save(mask_img, out_mask)
    print(f"[SAVED] {out_mask}")

    # QA plot using HO's own anatomical-like BG (use the cortical atlas mean as BG to avoid mismatched templates)
    # Minimal but sufficient: plot ROI over the same-space reference (here ref_img=cort/sub img); grayscale bg via max-prob projection
    # why: avoids MNI template variants (FSL vs 2009c) causing apparent misplacement
    # Make a 3D background (average across all atlas maps)
    bg_img = image.mean_img(ref_img)

    display = plotting.plot_roi(
        out_mask, bg_img=bg_img, display_mode="ortho",
        title=f"{roi} (HO space)", draw_cross=False, cmap="autumn", alpha=0.7
    )
    out_png = os.path.join(PLOT_DIR, f"{roi}.png")
    display.savefig(out_png, dpi=150)
    display.close()
    print(f"[PLOT] {out_png}")

print("[INFO] Done.")