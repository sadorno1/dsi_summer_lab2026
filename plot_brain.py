import os
import pickle
import numpy as np
import nibabel.freesurfer.io as fsio
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from load_data import ROI_NAME_TO_VALUE

# === CONFIG ===
condition_label = "ses-wk6_recap"
gsbs_out_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/GSBS_Results"
FIGURES_DIR = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/Figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# === 1. Load fsaverage mesh ===
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

# === 2. Load Schaefer-200 .annot files ===
lh_annot_path = "/project/ycleong/users/dsiSummerLab26/t-9sador/atlas_qc/lh.Schaefer2018_200Parcels_17Networks_order.annot"
rh_annot_path = "/project/ycleong/users/dsiSummerLab26/t-9sador/atlas_qc/rh.Schaefer2018_200Parcels_17Networks_order.annot"

lh_labels, lh_ctab, lh_names = fsio.read_annot(lh_annot_path)
rh_labels, rh_ctab, rh_names = fsio.read_annot(rh_annot_path)

# === 3. Load GSBS results ===
print("[INFO] Loading GSBS results...")
with open(os.path.join(gsbs_out_dir, f"students_results_{condition_label}.pkl"), "rb") as f:
    students_results = pickle.load(f)
with open(os.path.join(gsbs_out_dir, f"experts_results_{condition_label}.pkl"), "rb") as f:
    experts_results = pickle.load(f)

# === 4. Map results onto surface vertices ===
def build_vertex_value_array(labels, names, gsbs_results, roi_name_to_value):
    surf_values = np.full(len(labels), np.nan)
    for label_int in np.unique(labels):
        name = names[label_int]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if name in roi_name_to_value:
            roi_idx = roi_name_to_value[name]
            if roi_idx in gsbs_results:
                surf_values[labels == label_int] = gsbs_results[roi_idx]
    return surf_values

lh_students = build_vertex_value_array(lh_labels, lh_names, students_results, ROI_NAME_TO_VALUE)
rh_students = build_vertex_value_array(rh_labels, rh_names, students_results, ROI_NAME_TO_VALUE)
lh_experts  = build_vertex_value_array(lh_labels, lh_names, experts_results,  ROI_NAME_TO_VALUE)
rh_experts  = build_vertex_value_array(rh_labels, rh_names, experts_results,  ROI_NAME_TO_VALUE)

CMAP = 'YlGnBu'

def plot_group_brain(lh_vals, rh_vals, group_name):
    # Per-group color scale at 10th-90th percentile
    vals = np.concatenate([lh_vals[~np.isnan(lh_vals)], rh_vals[~np.isnan(rh_vals)]])
    vmin, vmax = np.percentile(vals, [10, 90])
    print(f"[INFO] {group_name} color scale: vmin={vmin:.1f}, vmax={vmax:.1f}")

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, wspace=0.02, hspace=0.02)

    views = [
        (0, 0, fsaverage['infl_left'],  lh_vals, 'left',  'lateral', fsaverage['sulc_left']),
        (0, 1, fsaverage['infl_right'], rh_vals, 'right', 'lateral', fsaverage['sulc_right']),
        (1, 0, fsaverage['infl_left'],  lh_vals, 'left',  'medial',  fsaverage['sulc_left']),
        (1, 1, fsaverage['infl_right'], rh_vals, 'right', 'medial',  fsaverage['sulc_right']),
    ]

    for row, col, mesh, vals_view, hemi, view, sulc in views:
        ax = fig.add_subplot(gs[row, col], projection='3d')
        plotting.plot_surf_roi(
            mesh, roi_map=vals_view, hemi=hemi, view=view,
            bg_map=sulc, cmap=CMAP, vmin=vmin, vmax=vmax,
            axes=ax, colorbar=False
        )

    cax = fig.add_axes([0.92, 0.18, 0.015, 0.65])
    sm = cm.ScalarMappable(cmap=CMAP, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Number of Events', fontsize=11, labelpad=10)

    fig.text(0.46, 0.02, group_name, ha='center', fontsize=14, fontweight='bold')
    plt.suptitle(
        f"GSBS Event Counts — {group_name} ({condition_label})\nColor scale: {vmin:.0f}–{vmax:.0f} events (10th–90th percentile)",
        fontsize=12, y=0.98
    )

    out_path = os.path.join(FIGURES_DIR, f"gsbs_event_brainmap_{condition_label}_{group_name.lower()}_10_90.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path}")

plot_group_brain(lh_students, rh_students, "Students")
plot_group_brain(lh_experts,  rh_experts,  "Experts")

print("[DONE]")