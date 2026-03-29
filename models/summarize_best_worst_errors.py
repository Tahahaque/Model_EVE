# summarize_best_worst_errors.py
import os
import numpy as np

from analyze_best_worst_with_eyes import collect_frame_records

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
    os.makedirs("results", exist_ok=True)

    # reuse the same logic to collect per-frame errors
    records = collect_frame_records(root, seq_len=30, max_seqs=200)
    print(f"Collected {len(records)} records")

    # sort by refine error
    rec_sorted = sorted(records, key=lambda r: r["err_refine"])

    k = 5
    best = rec_sorted[:k]
    worst = rec_sorted[-k:]

    def avg_err(rs, key):
        return float(np.mean([r[key] for r in rs]))

    best_refine = avg_err(best, "err_refine")
    best_eye    = avg_err(best, "err_eyenet")
    worst_refine = avg_err(worst, "err_refine")
    worst_eye    = avg_err(worst, "err_eyenet")

    # print nicely for the thesis
    print("=== Average angular error on best and worst 5 frames ===")
    print(f"Best 5 frames:  refine = {best_refine:.2f}°, EyeNet-GRU = {best_eye:.2f}°")
    print(f"Worst 5 frames: refine = {worst_refine:.2f}°, EyeNet-GRU = {worst_eye:.2f}°")

    # save as CSV so you can drop it into a table
    import csv
    out_csv = "results/best_worst_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "model", "mean_error_deg"])
        writer.writerow(["best5", "GazeRefineNet", best_refine])
        writer.writerow(["best5", "EyeNetGRU", best_eye])
        writer.writerow(["worst5", "GazeRefineNet", worst_refine])
        writer.writerow(["worst5", "EyeNetGRU", worst_eye])
    print("Saved summary to", out_csv)

if __name__ == "__main__":
    main()