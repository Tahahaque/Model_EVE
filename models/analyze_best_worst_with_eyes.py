# analyze_best_worst_with_eyes.py
import os
import glob
import math

import numpy as np
import cv2
import h5py
import torch
import matplotlib.pyplot as plt

from dataset_gazerefine_yawpitch import GazeRefineYawPitchDataset
from model_gazerefine_yawpitch_gru import GazeRefineYawPitchGRU
from model_gru_temp import EyeNetGRU, mean_angular_error_deg

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def to_device():
    if DEVICE == "cuda":
        return torch.device("cuda")
    if DEVICE == "mps":
        return torch.device("mps")
    return torch.device("cpu")

DEVICE_T = to_device()

# ---------- helpers to load frames ----------

def load_screen_frame(step_dir, frame_idx):
    path = os.path.join(step_dir, "screen.128x72.mp4")
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame  # H,W,3

def load_eye_frame(step_dir, frame_idx, img_size=(64, 64)):
    npy_dir = os.path.join(step_dir, "basler_eyes_npy")
    fname = os.path.join(npy_dir, f"{int(frame_idx):06d}.npy")
    if not os.path.isfile(fname):
        return None
    gray = np.load(fname)  # H0,W0
    gray = cv2.resize(gray, img_size, interpolation=cv2.INTER_AREA)
    return gray  # H,W

def simple_face_crop(screen_img):
    # very simple: center crop square; adjust if you have better annotations
    h, w, _ = screen_img.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = screen_img[y0:y0+side, x0:x0+side]
    return crop

# ---------- load trained models ----------

def load_refine_model():
    model = GazeRefineYawPitchGRU(
        cnn_out_channels=64,
        gru_hidden_size=128,
        gru_layers=1,
    )
    ckpt = torch.load("checkpoints/gazerefine_yawpitch_gru_best.pth",
                      map_location=DEVICE_T)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE_T)
    model.eval()
    return model

def load_eyenet_model():
    model = EyeNetGRU(out_gaze_dim=2, hidden_dim=256)
    ckpt = torch.load("checkpoints/eyenet_gru_epoch5.pth",
                      map_location=DEVICE_T)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE_T)
    model.eval()
    return model

# ---------- collect per-frame errors (refine) ----------

@torch.no_grad()
def collect_frame_records(root, seq_len=30, max_seqs=200):
    ds = GazeRefineYawPitchDataset(
        root=root,
        split="val01",
        seq_len=seq_len,
        step_stride=seq_len,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    refine = load_refine_model()
    eyenet = load_eyenet_model()

    records = []
    seq_count = 0

    for batch_idx, batch in enumerate(loader):
        screen_seq = batch["screen_seq"].to(DEVICE_T)           # (1,T,3,72,128)
        eyenet_seq = batch["eyenet_seq"].to(DEVICE_T)           # (1,T,2)
        gt_seq     = batch["gt_yawpitch_seq"].to(DEVICE_T)      # (1,T,2)
        step_dir   = batch["step_dir"][0]
        start_idx  = batch["start_idx"].item()

        # refine prediction
        pred_refine = refine(screen_seq, eyenet_seq)[0]         # (1,T,2)
        pred_refine = pred_refine.squeeze(0).cpu()              # (T,2)

        # eyenet predictions on the same eye sequence
        # we need the eye frames in the same order
        # -> EVEEyeSequenceDataset already uses basler_eyes_npy, but here
        #    we just infer directly from the numpy eye frames
        # load eyes for this step and slice same indices
        # (re-use your loader function if you prefer)
        eyes_all = []
        npy_dir = os.path.join(step_dir, "basler_eyes_npy")
        for t in range(seq_len):
            frame_idx = start_idx + t
            fname = os.path.join(npy_dir, f"{int(frame_idx):06d}.npy")
            if not os.path.isfile(fname):
                break
            gray = np.load(fname)
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            img = gray.astype(np.float32) / 255.0
            eyes_all.append(img[None, ...])  # (1,H,W)

        if len(eyes_all) == 0:
            continue

        eyes = np.stack(eyes_all, axis=0)                        # (T,1,H,W)
        eyes_t = torch.from_numpy(eyes).unsqueeze(0).to(DEVICE_T)  # (1,T,1,H,W)
        pred_eye, _ = eyenet(eyes_t)                             # (1,T,2)
        pred_eye = pred_eye.squeeze(0).cpu()                     # (T,2)

        gt_flat = gt_seq.squeeze(0).cpu()                        # (T,2)

        for t in range(pred_refine.size(0)):
            pred_r = pred_refine[t:t+1]
            pred_e = pred_eye[t:t+1]
            gt_t   = gt_flat[t:t+1]

            err_refine = mean_angular_error_deg(pred_r, gt_t)
            err_eye    = mean_angular_error_deg(pred_e, gt_t)

            records.append({
                "step_dir": step_dir,
                "frame_idx": start_idx + t,
                "err_refine": err_refine,
                "err_eyenet": err_eye,
            })

        seq_count += 1
        if max_seqs is not None and seq_count >= max_seqs:
            break

    return records

# ---------- plotting ----------

def plot_best_worst(records, out_path="results/best_worst_face_eye.png", k=5):
    # sort by refine error
    records_sorted = sorted(records, key=lambda r: r["err_refine"])
    best = records_sorted[:k]
    worst = records_sorted[-k:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # figure: 4 rows (screen, face, eye, bar text), k columns
    rows = 3
    fig, axes = plt.subplots(rows, 2*k, figsize=(3*2*k, 3*rows))

    # helper to fill a column-group (best on left block, worst on right block)
    def fill_group(rec, row_offset, col_idx, title_prefix):
        screen = load_screen_frame(rec["step_dir"], rec["frame_idx"])
        if screen is None:
            return
        face = simple_face_crop(screen)
        eye  = load_eye_frame(rec["step_dir"], rec["frame_idx"])

        # row 0: screen
        ax = axes[0, col_idx]
        ax.imshow(screen)
        ax.axis("off")
        ax.set_title(f"{title_prefix}\nref={rec['err_refine']:.2f}°, eye={rec['err_eyenet']:.2f}°")

        # row 1: face crop
        ax = axes[1, col_idx]
        ax.imshow(face)
        ax.axis("off")

        # row 2: eye image (grayscale)
        ax = axes[2, col_idx]
        if eye is not None:
            ax.imshow(eye, cmap="gray")
        ax.axis("off")

    # fill best on left block
    for i, r in enumerate(best):
        fill_group(r, 0, i, f"Best {i+1}")

    # fill worst on right block
    for i, r in enumerate(worst):
        fill_group(r, 0, i + k, f"Worst {i+1}")

    fig.suptitle("Refine vs EyeNet-GRU: best and worst frames (screen, face, eye)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
    print("Collecting per-frame records...")
    records = collect_frame_records(root, seq_len=30, max_seqs=200)
    print(f"Collected {len(records)} records.")
    print("Plotting...")
    plot_best_worst(records, out_path="results/best_worst_face_eye.png", k=5)
    print("Saved to results/best_worst_face_eye.png")

if __name__ == "__main__":
    main()