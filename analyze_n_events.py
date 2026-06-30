import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from load_data import ROI_NAME_TO_VALUE
ROI_VALUE_TO_NAME = {v: k for k, v in ROI_NAME_TO_VALUE.items()}

# === CONFIG ===
RESULTS_DIR = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/GSBS_Results"
OUTPUT_DIR  = "/project/ycleong/users/dsiSummerLab26/CS_Learning/pipeline_results_recap/Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITION = "ses-wk6_recap"

# === LOAD ===
with open(os.path.join(RESULTS_DIR, f"students_results_{CONDITION}.pkl"), "rb") as f:
    students_results = pickle.load(f)
with open(os.path.join(RESULTS_DIR, f"experts_results_{CONDITION}.pkl"), "rb") as f:
    experts_results = pickle.load(f)

def nevents(obj):
    if hasattr(obj, "nstates"):
        return int(obj.nstates)
    return int(obj)

def counts_dict(results):
    return {roi: nevents(obj) for roi, obj in results.items() if obj is not None}

stu_counts = counts_dict(students_results)
exp_counts = counts_dict(experts_results)

print(f"[INFO] students: {len(stu_counts)} ROIs covered")
print(f"[INFO] experts:  {len(exp_counts)} ROIs covered")

# === 1. HISTOGRAMS OF EVENT COUNTS ACROSS ROIS ===
stu_vals = np.array(list(stu_counts.values()))
exp_vals = np.array(list(exp_counts.values()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].hist(stu_vals, bins=30, color='steelblue', edgecolor='black')
axes[0].set_title(f"Students  (n_ROI={len(stu_vals)})\nmean={stu_vals.mean():.1f}, "
                  f"median={np.median(stu_vals):.1f}, range=[{stu_vals.min()}, {stu_vals.max()}]")
axes[0].set_xlabel("Event count (GSBS nstates)")
axes[0].set_ylabel("Number of ROIs")

axes[1].hist(exp_vals, bins=30, color='darkorange', edgecolor='black')
axes[1].set_title(f"Experts  (n_ROI={len(exp_vals)})\nmean={exp_vals.mean():.1f}, "
                  f"median={np.median(exp_vals):.1f}, range=[{exp_vals.min()}, {exp_vals.max()}]")
axes[1].set_xlabel("Event count (GSBS nstates)")

plt.suptitle(f"Distribution of GSBS event counts across 200 ROIs  ({CONDITION})")
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, f"event_count_hist_{CONDITION}.png")
plt.savefig(out1, dpi=300)
plt.close()
print(f"[SAVED] {out1}")

# === 2a. SORTED BAR CHART (shape of distribution, rank-ordered) ===
stu_sorted = sorted(stu_counts.items(), key=lambda x: -x[1])
exp_sorted = sorted(exp_counts.items(), key=lambda x: -x[1])

fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharey=True)

axes[0].bar(range(len(stu_sorted)), [v for _, v in stu_sorted], color='steelblue', width=1.0)
axes[0].set_title(f"Students — ROIs sorted by event count, n={len(stu_sorted)}")
axes[0].set_ylabel("GSBS event count")

axes[1].bar(range(len(exp_sorted)), [v for _, v in exp_sorted], color='darkorange', width=1.0)
axes[1].set_title(f"Experts — ROIs sorted by event count, n={len(exp_sorted)}")
axes[1].set_ylabel("GSBS event count")
axes[1].set_xlabel("ROI rank (sorted, not anatomical index)")

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, f"roi_sorted_event_counts_{CONDITION}.png")
plt.savefig(out2, dpi=300)
plt.close()
print(f"[SAVED] {out2}")

# === 2b. BAR CHART BY ANATOMICAL ROI INDEX (keeps ROI identity) ===
fig, axes = plt.subplots(2, 1, figsize=(22, 10), sharey=True)

stu_rois = sorted(stu_counts.keys())
stu_y    = [stu_counts[r] for r in stu_rois]

exp_rois = sorted(exp_counts.keys())
exp_y    = [exp_counts[r] for r in exp_rois]

axes[0].bar(stu_rois, stu_y, color='steelblue', width=0.9)
axes[0].set_title("Students — event count per ROI (anatomical index, 1-200)")
axes[0].set_ylabel("GSBS event count")

axes[1].bar(exp_rois, exp_y, color='darkorange', width=0.9)
axes[1].set_title("Experts — event count per ROI (anatomical index, 1-200)")
axes[1].set_ylabel("GSBS event count")
axes[1].set_xlabel("ROI index (Schaefer-200 / 17Networks)")

plt.tight_layout()
out2b = os.path.join(OUTPUT_DIR, f"roi_index_event_counts_{CONDITION}.png")
plt.savefig(out2b, dpi=300)
plt.close()
print(f"[SAVED] {out2b}")

# === 3. VIS vs DEFAULT NETWORK COMPARISON ===

# Networks in predicted hierarchy order (fastest -> slowest timescale)
# i.e. most events on the left, fewest on the right if hierarchy is intact
NETWORK_ORDER = [
    "VisCent", "VisPeri",
    "SomMotA", "SomMotB",
    "DorsAttnA", "DorsAttnB",
    "SalVentAttnA", "SalVentAttnB",   # check naming -- could be SalVentnA in your atlas labels
    "ContA", "ContB", "ContC",
    "DefaultA", "DefaultB", "DefaultC",
    "LimbicA", "LimbicB",          
    "TempPar",
]


# === Map each ROI to its network ===
def get_network(roi_name):
    # ROI names look like "17Networks_LH_VisCent_ExStr_1"
    # network identifier is the third underscore-separated token
    parts = roi_name.split("_")
    if len(parts) < 3:
        return None
    return parts[2]

ROI_TO_NETWORK = {}
for name, idx in ROI_NAME_TO_VALUE.items():
    net = get_network(name)
    if net is not None:
        ROI_TO_NETWORK[idx] = net

# print which network labels actually exist in your atlas (sanity check)
unique_nets = sorted(set(ROI_TO_NETWORK.values()))
print(f"[INFO] Network labels found in atlas: {unique_nets}")

# === Compute per-network event count distributions ===
def per_network_counts(counts):
    result = {net: [] for net in unique_nets}
    for roi, n in counts.items():
        net = ROI_TO_NETWORK.get(roi)
        if net is not None:
            result[net].append(n)
    return {net: np.array(vals) for net, vals in result.items() if len(vals) > 0}

stu_net = per_network_counts(stu_counts)
exp_net = per_network_counts(exp_counts)

# Use only the networks that actually exist in the atlas, in hierarchy order
plot_networks = [n for n in NETWORK_ORDER if n in stu_net and n in exp_net]
# also tack on any unexpected leftover networks at the end so nothing gets dropped silently
leftover = [n for n in unique_nets if n not in plot_networks]
plot_networks = plot_networks + leftover

# === Plot: side-by-side bars (students vs experts) per network, in hierarchy order ===
fig, ax = plt.subplots(figsize=(14, 6))

xpos   = np.arange(len(plot_networks))
width  = 0.38

stu_means = [stu_net[n].mean() for n in plot_networks]
stu_sems  = [stu_net[n].std() / np.sqrt(len(stu_net[n])) for n in plot_networks]
exp_means = [exp_net[n].mean() for n in plot_networks]
exp_sems  = [exp_net[n].std() / np.sqrt(len(exp_net[n])) for n in plot_networks]

ax.bar(xpos - width/2, stu_means, width, yerr=stu_sems, label='Students',
       color='steelblue', edgecolor='black', capsize=3, alpha=0.85)
ax.bar(xpos + width/2, exp_means, width, yerr=exp_sems, label='Experts',
       color='darkorange', edgecolor='black', capsize=3, alpha=0.85)

# overlay individual ROI dots
for i, net in enumerate(plot_networks):
    jitter_s = np.random.uniform(-0.08, 0.08, size=len(stu_net[net]))
    jitter_e = np.random.uniform(-0.08, 0.08, size=len(exp_net[net]))
    ax.scatter(xpos[i] - width/2 + jitter_s, stu_net[net], color='black', alpha=0.35, s=8, zorder=3)
    ax.scatter(xpos[i] + width/2 + jitter_e, exp_net[net], color='black', alpha=0.35, s=8, zorder=3)

ax.set_xticks(xpos)
ax.set_xticklabels(plot_networks, rotation=45, ha='right')
ax.set_ylabel("GSBS event count")
ax.set_title(f"Event count per network ({CONDITION})\n"
             "Networks ordered by predicted hierarchy (left=fast/sensory, right=slow/transmodal)")
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, f"event_count_per_network_{CONDITION}.png")
plt.savefig(out, dpi=300)
plt.close()
print(f"[SAVED] {out}")

# === Quantitative test: does mean event count follow the predicted hierarchy order? ===
# Spearman correlation between predicted hierarchy rank (1=fastest) and observed mean event count
# Negative correlation expected: higher hierarchy rank (further from sensory) -> fewer events

# build hierarchy ranks, excluding limbic/temppar which don't fit the gradient
hierarchy_networks = [n for n in plot_networks if n not in ("LimbicA", "LimbicB", "TempPar")]
hierarchy_ranks    = np.arange(len(hierarchy_networks))  # 0 = most sensory, high = most transmodal

stu_means_h = [stu_net[n].mean() for n in hierarchy_networks]
exp_means_h = [exp_net[n].mean() for n in hierarchy_networks]

rho_stu, p_stu = spearmanr(hierarchy_ranks, stu_means_h)
rho_exp, p_exp = spearmanr(hierarchy_ranks, exp_means_h)

print("\n=== Hierarchy gradient test (rank vs mean event count) ===")
print("Negative rho = expected hierarchy direction (sensory has more events)")
print(f"  STUDENTS  rho = {rho_stu:+.3f}  p = {p_stu:.3g}")
print(f"  EXPERTS   rho = {rho_exp:+.3f}  p = {p_exp:.3g}")

# === Detailed table per network ===
print(f"\n{'Network':<14} {'n_ROI':>5}  {'Stu mean ± SEM':>16}  {'Exp mean ± SEM':>16}")
for n in plot_networks:
    sm, ss = stu_net[n].mean(), stu_net[n].std() / np.sqrt(len(stu_net[n]))
    em, es = exp_net[n].mean(), exp_net[n].std() / np.sqrt(len(exp_net[n]))
    print(f"{n:<14} {len(stu_net[n]):>5}  {sm:>7.2f} ± {ss:>5.2f}  {em:>7.2f} ± {es:>5.2f}")
# === 4. FULL ROI TABLE: index, name, student count, expert count ===
print(f"\n{'ROI':>4}  {'Name':<45} {'Stu':>5} {'Exp':>5}")
for roi in range(1, 201):
    name = ROI_VALUE_TO_NAME.get(roi, "???")
    s = stu_counts.get(roi, None)
    e = exp_counts.get(roi, None)
    print(f"{roi:>4}  {name:<45} {str(s):>5} {str(e):>5}")

# === 5. VISUAL-TO-TRANSMODAL HIERARCHY PLOT (V1 ... vmpfc), students vs experts ===
# Manually curated ROI groups (Schaefer-200/17Networks indices), conservative + liberal sets.
# Order = predicted hierarchy: sensory (left) -> transmodal (right), matching the reference figure.

REGION_ORDER = ["V1", "V2, V3", "V4, V5", "IPS", "AG", "dmpfc", "vmpfc"]

REGIONS_CONSERVATIVE = {
    "V1":     [3, 103, 109],
    "V2, V3": [12, 104, 6, 106],
    "V4, V5": [1, 101, 31, 5, 105, 102],
    "IPS":    [136, 166, 165, 60, 59, 32],
    "AG":     [75, 184, 194],
    "dmpfc":  [90, 89, 192],
    "vmpfc":  [80, 159],
}

# extra ROIs added only in the liberal version
REGIONS_LIBERAL_EXTRA = {
    "V1":     [10],
    "V2, V3": [2, 8, 4, 112],
    "V4, V5": [30, 132],
    "IPS":    [135, 58, 36],
    "AG":     [68, 87, 173],
    "dmpfc":  [189],
    "vmpfc":  [52],
}

REGIONS_LIBERAL = {
    r: REGIONS_CONSERVATIVE[r] + REGIONS_LIBERAL_EXTRA[r] for r in REGION_ORDER
}

REGION_COLORS = {
    "V1":     "#1b1b8f",
    "V2, V3": "#7a2b8a",
    "V4, V5": "#2c7d80",
    "IPS":    "#3ec0cf",
    "AG":     "#4f9fe0",
    "dmpfc":  "#f0922f",
    "vmpfc":  "#f3cf3f",
}

ADD_ERRORBARS = False  # reference figure has none; flip to True for SEM across ROIs


def region_mean(counts, roi_list):
    """sum(event counts in roi_list) / n, ignoring ROIs missing from this condition's results"""
    vals    = [counts[r] for r in roi_list if r in counts]
    missing = [r for r in roi_list if r not in counts]
    if not vals:
        return np.nan, 0.0, [], missing
    vals = np.array(vals, dtype=float)
    sem  = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return vals.mean(), sem, list(vals.astype(int)), missing


def print_roi_breakdown(version_name, region_to_rois):
    """Print each ROI's individual count + atlas name, per region, students vs experts.
    Lets you eyeball outliers and check ROI->region assignment by name instead of just index."""
    print(f"\n=== ROI-LEVEL BREAKDOWN — {version_name.upper()} ({CONDITION}) ===")
    for r in REGION_ORDER:
        print(f"\n-- {r} --")
        print(f"  {'ROI':>4}  {'Atlas name':<40} {'Stu':>5} {'Exp':>5}")
        for roi in region_to_rois[r]:
            name = ROI_VALUE_TO_NAME.get(roi, "??? (not in atlas)")
            s    = stu_counts.get(roi, "missing")
            e    = exp_counts.get(roi, "missing")
            print(f"  {roi:>4}  {name:<40} {str(s):>5} {str(e):>5}")


def make_hierarchy_figure(version_name, region_to_rois):
    colors = [REGION_COLORS[r] for r in REGION_ORDER]
    x      = np.arange(len(REGION_ORDER))

    stu = [region_mean(stu_counts, region_to_rois[r]) for r in REGION_ORDER]
    exp = [region_mean(exp_counts, region_to_rois[r]) for r in REGION_ORDER]
    stu_means, stu_sems = [s[0] for s in stu], [s[1] for s in stu]
    exp_means, exp_sems = [e[0] for e in exp], [e[1] for e in exp]

    print(f"\n=== VISUAL HIERARCHY — {version_name.upper()} ({CONDITION}) ===")
    print(f"{'Region':<8} {'nROI':>4} {'Students':>9} {'Experts':>9}   ROIs used")
    for i, r in enumerate(REGION_ORDER):
        used    = stu[i][2]
        missing = stu[i][3]
        miss_s  = f"  MISSING {missing}" if missing else ""
        print(f"{r:<8} {len(used):>4} {stu_means[i]:>9.2f} {exp_means[i]:>9.2f}   "
              f"{region_to_rois[r]}{miss_s}")

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)

    se_stu = stu_sems if ADD_ERRORBARS else None
    se_exp = exp_sems if ADD_ERRORBARS else None

    axes[0].bar(x, stu_means, yerr=se_stu, color=colors, edgecolor='black', width=0.8, capsize=3)
    axes[0].text(0.97, 0.82, "Students", transform=axes[0].transAxes, ha='right', fontsize=14)

    axes[1].bar(x, exp_means, yerr=se_exp, color=colors, edgecolor='black', width=0.8, capsize=3)
    axes[1].text(0.97, 0.82, "Experts", transform=axes[1].transAxes, ha='right', fontsize=14)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(REGION_ORDER, rotation=45, ha='right', style='italic')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.02)

    fig.supylabel("Number of Events")
    fig.suptitle(f"GSBS event count, visual->transmodal hierarchy\n{version_name} ({CONDITION})",
                 fontsize=11)
    plt.tight_layout()

    out_h = os.path.join(OUTPUT_DIR, f"visual_hierarchy_{version_name}_{CONDITION}.png")
    plt.savefig(out_h, dpi=300)
    plt.close()
    print(f"[SAVED] {out_h}")


print_roi_breakdown("conservative", REGIONS_CONSERVATIVE)
print_roi_breakdown("liberal", REGIONS_LIBERAL)

make_hierarchy_figure("conservative", REGIONS_CONSERVATIVE)
make_hierarchy_figure("liberal", REGIONS_LIBERAL)