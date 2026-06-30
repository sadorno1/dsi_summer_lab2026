import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main(data_dir):
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} not found.")
        sys.exit(1)

    # Initialize a 3-level deep nested dictionary so multiple files don't overwrite each other
    info = defaultdict(lambda: defaultdict(dict))  # info[subject][session][file] = rois_info
    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "sub-*")))

    for sub in subject_dirs:
        sub_id = os.path.basename(sub)
        ses_dirs = sorted(glob.glob(os.path.join(sub, "ses-*")))
        for ses in ses_dirs:
            ses_id = os.path.basename(ses)
            npz_files = sorted(glob.glob(os.path.join(ses, "*_parcels.npz")))
            for fpath in npz_files:
                fname = os.path.basename(fpath)
                data = np.load(fpath)
                rois = {k: data[k] for k in data.files}
                
                # Store shape and basic stats for each ROI
                rois_info = {}
                for roi, arr in rois.items():
                    if arr.size == 0:
                        rois_info[roi] = {'shape': arr.shape, 'mean': np.nan, 'std': np.nan}
                    else:
                        rois_info[roi] = {
                            'shape': arr.shape,
                            'mean': np.mean(arr),
                            'std': np.std(arr),
                            'min': np.min(arr),
                            'max': np.max(arr),
                        }
                info[sub_id][ses_id][fname] = rois_info

    # Print summary
    print("\n=== PARSED DATA SUMMARY ===\n")
    for sub, ses_dict in info.items():
        print(f"Subject: {sub}")
        for ses, file_dict in ses_dict.items():
            print(f"  Session: {ses}")
            for fname, rois in file_dict.items():
                print(f"    File: {fname}")
                for roi, stats in rois.items():
                    shape = stats['shape']
                    mean = stats['mean']
                    std = stats['std']
                    print(f"      ROI {roi}: shape {shape}, mean {mean:.4f}, std {std:.4f}")
        print()

    # Calculate total timepoints for the plot across all files per subject
    timepoints = {}
    for sub, ses_dict in info.items():
        total_t = 0
        for ses, file_dict in ses_dict.items():
            for fname, rois in file_dict.items():
                for roi, stats in rois.items():
                    if len(stats['shape']) == 2:
                        total_t += stats['shape'][1]
                        break 
        timepoints[sub] = total_t

    # --- NEW: Task/Segment Duration breakdown for a single subject ---
    if info:
        print("=== STIMULUS LENGTH BREAKDOWN (TIMEPOINTS) ===")
        # Grabbing the first subject available in the parsed data structure
        sample_sub = list(info.keys())[0]
        print(f"Target Subject: {sample_sub}\n")
        
        for ses, file_dict in info[sample_sub].items():
            print(f"Session: {ses}")
            for fname, rois in file_dict.items():
                for roi, stats in rois.items():
                    if len(stats['shape']) == 2:
                        tp_length = stats['shape'][1]
                        # Clean up the long string filename to make the output easily readable
                        short_name = fname.split('_space-')[0]
                        print(f"  - {short_name}: {tp_length} timepoints (TRs)")
                        break
        print("==============================================")

    # Plot
    if timepoints:
        plt.figure(figsize=(12, 6))
        subjects = list(timepoints.keys())
        tps = list(timepoints.values())
        plt.bar(subjects, tps)
        plt.xticks(rotation=90)
        plt.xlabel("Subject")
        plt.ylabel("Total timepoints (all files)")
        plt.title("Total number of TRs per subject")
        plt.tight_layout()
        plt.savefig("timepoints_per_subject.png", dpi=150)
        print("[INFO] Saved timepoints plot as timepoints_per_subject.png")

    # Save a CSV with detailed stats 
    import csv
    csv_file = "parcellated_stats.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "session", "file", "roi", "voxels", "timepoints",
                         "mean", "std", "min", "max"])
        for sub, ses_dict in info.items():
            for ses, file_dict in ses_dict.items():
                for fname, rois in file_dict.items():
                    for roi, stats in rois.items():
                        if len(stats['shape']) == 2:
                            vox, tp = stats['shape']
                        else:
                            vox, tp = stats['shape'][0], np.nan
                        writer.writerow([sub, ses, fname, roi, vox, tp,
                                         stats['mean'], stats['std'],
                                         stats.get('min', np.nan), stats.get('max', np.nan)])
    print(f"[INFO] Saved detailed stats to {csv_file}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/project/ycleong/users/dsiSummerLab26/CS_Learning/parcellated_data"
    main(path)