# analyze_best_worst_images.py
import os
import glob
import math

import numpy as np
import cv2
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model_gazerefine_yawpitch_gru import GazeRefineYawPitchGRU
from dataset_gazerefine_yawpitch import GazeRefineYawPitchDataset
from model_gru_temp import mean_angular_error_deg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_screen_frame(step_dir, frame_idx):
    """Load one RGB frame from screen.128x72.mp4 at index frame_idx."""
    vid_path = os.path.join(step_dir, "screen.128x72.mp4")
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def collect_per_frame_errors(root, seq_len=30, max_items=200):
    ds = GazeRefineYawPitchDataset(
        root=root,
        split="val01",
        seq_len=seq_len,
        step_stride=seq_len,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = GazeRefineYawPitchGRU(
        cnn_out_channels=64,
        gru_hidden_size=128,
        gru_layers=1,
    )
    ckpt = torch.load("checkpoints/gazerefine_yawpitch_gru_best.pth",
                      map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    records = []
    n_items = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            screen_seq = batch["screen_seq"].to(DEVICE)         # (1,T,3,72,128)
            eyenet_seq = batch["eyenet_seq"].to(DEVICE)         # (1,T,2)
            gt_seq = batch["gt_yawpitch_seq"].to(DEVICE)        # (1,T,2)

            pred_seq = model(screen_seq, eyenet_seq)            # (1,T,2)
            T = pred_seq.size(1)

            # compute per-frame angular error
            pred_flat = pred_seq.view(T, 2).cpu()
            gt_flat = gt_seq.view(T, 2).cpu()

            v_pred = torch.stack([
                torch.cos(gt_flat[:,1]) * torch.sin(gt_flat[:,0]),
                torch.sin(gt_flat[:,1]),
                torch.cos(gt_flat[:,1]) * torch.cos(gt_flat[:,0]),
            ], dim=1)
            # use helper to compute per frame
            # easier: reuse mean_angular_error_deg on singletons
            for t in range(T):
                ang = mean_angular_error_deg(
                    pred_flat[t:t+1],
                    gt_flat[t:t+1],
                )
                # identify which step_dir this comes from
                step_i, start = ds.index[i]
                frame_idx = start + t
                step_dir = ds.step_dirs[step_i]
                records.append({
                    "error_deg": ang,
                    "step_dir": step_dir,
                    "frame_idx": frame_idx,
                })

            n_items += 1
            if max_items is not None and n_items >= max_items:
                break

    return records

def plot_best_worst(records, out_path="results/best_worst_frames.png", k=3):
    records_sorted = sorted(records, key=lambda r: r["error_deg"])
    best = records_sorted[:k]
    worst = records_sorted[-k:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 2 rows: best (top), worst (bottom)
    fig, axes = plt.subplots(2, k, figsize=(4*k, 6))

    for i, r in enumerate(best):
        img = load_screen_frame(r["step_dir"], r["frame_idx"])
        ax = axes[0, i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Best {i+1}\nerr={r['error_deg']:.2f}°")

    for i, r in enumerate(worst):
        img = load_screen_frame(r["step_dir"], r["frame_idx"])
        ax = axes[1, i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Worst {i+1}\nerr={r['error_deg']:.2f}°")

    fig.suptitle("GazeRefine-GRU: best vs worst frames")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
    records = collect_per_frame_errors(root, seq_len=30, max_items=200)
    print(f"Collected {len(records)} frame errors")
    plot_best_worst(records, out_path="results/best_worst_frames.png", k=3)
    print("Saved best/worst figure to results/best_worst_frames.png")

if __name__ == "__main__":
    main()