import os
from nilearn import datasets, plotting

# === PATHS ===
output_dir = "/project/ycleong/users/dsiSummerLab26/t-9sador/atlas_qc"
os.makedirs(output_dir, exist_ok=True)

# === LOAD SCHAEFER 200 / 17-NETWORK ATLAS ===
print("[INFO] Fetching Schaefer 200 / 17-network atlas...")
schaefer = datasets.fetch_atlas_schaefer_2018(
    n_rois=200, yeo_networks=17, resolution_mm=2
)
print(f"[INFO] Atlas loaded with {len(schaefer.labels)} parcels")

# === LOAD MNI152 BACKGROUND ===
mni = datasets.load_mni152_template(resolution=2)
print("[INFO] Loaded MNI152 2mm template for background.")

# === PLOT 1: ORTHO VIEW (all parcels colored) ===
print("[INFO] Plotting ortho view...")
display = plotting.plot_roi(
    schaefer.maps,
    bg_img=mni,
    display_mode="ortho",
    title="Schaefer 200 / 17-network atlas (ortho)",
    draw_cross=False,
    cmap="tab20",
    alpha=0.7,
)
ortho_path = os.path.join(output_dir, "schaefer200_17net_ortho.png")
display.savefig(ortho_path, dpi=150)
display.close()
print(f"    [SAVED] {ortho_path}")

# === PLOT 2: AXIAL SLICES (more comprehensive coverage) ===
print("[INFO] Plotting axial slices...")
display = plotting.plot_roi(
    schaefer.maps,
    bg_img=mni,
    display_mode="z",
    cut_coords=8,
    title="Schaefer 200 / 17-network atlas (axial)",
    draw_cross=False,
    cmap="tab20",
    alpha=0.7,
)
axial_path = os.path.join(output_dir, "schaefer200_17net_axial.png")
display.savefig(axial_path, dpi=150)
display.close()
print(f"    [SAVED] {axial_path}")

print(f"[DONE] QC plots saved to: {output_dir}")