# plot correlation between number of events between students and experts

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# === CONFIG ===
condition_label = "ses-wk6_recap"
gsbs_out_dir = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/GSBS_Results"
FIGURES_DIR = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/Figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# === LOAD RESULTS ===
with open(os.path.join(gsbs_out_dir, f"students_results_{condition_label}.pkl"), "rb") as f:
    students_results = pickle.load(f)
with open(os.path.join(gsbs_out_dir, f"experts_results_{condition_label}.pkl"), "rb") as f:
    experts_results = pickle.load(f)

def nevents(obj):
    if hasattr(obj, "nstates"):
        return int(obj.nstates)
    return int(obj)

# === PANEL B: scatter + Spearman correlation ===
rois_common = sorted(set(students_results) & set(experts_results))
print(f"[INFO] {len(rois_common)} ROIs common to both groups")

x = np.array([nevents(students_results[r]) for r in rois_common], dtype=float)
y = np.array([nevents(experts_results[r])  for r in rois_common], dtype=float)

rho, p = spearmanr(x, y)
print(f"[RESULT] Spearman rho = {rho:.3f}, p = {p:.4g}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x, y, alpha=0.6, color='#4477AA', edgecolor='k', linewidth=0.3)

# Least-squares regression line
slope, intercept = np.polyfit(x, y, 1)
xs = np.array([x.min() - 1, x.max() + 1])
ax.plot(xs, slope * xs + intercept, color='#CC3311', linewidth=1.8,
        label=f'best fit: slope={slope:.2f}')

# Faint identity line for context
lims = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
ax.plot(lims, lims, 'k--', alpha=0.25, linewidth=1, label='y = x')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel('Number of Events (Students)')
ax.set_ylabel('Number of Events (Experts)')
ax.set_title(f"Spearman ρ = {rho:.3f}, p = {p:.3g}")
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, f"panel_b_spearman_{condition_label}.png")
plt.savefig(out_path, dpi=200)
print(f"[SAVED] {out_path}")