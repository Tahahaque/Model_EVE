import os
import glob
import h5py
import cv2
import numpy as np
import math
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -------------------- 1. Helpers --------------------

def list_step_dirs(root: str, split: str = "train", max_steps=None):
    """
    root: path to eve_dataset_2
    split: 'train' | 'val' | 'test'
    """
    if split == "train":
        pat = os.path.join(root, "train*", "step*")
    elif split == "val":
        pat = os.path.join(root, "val*", "step*")
    elif split == "test":
        pat = os.path.join(root, "test*", "step*")
    else:
        raise ValueError(split)
    dirs = sorted(glob.glob(pat))
    if max_steps is not None:
        dirs = dirs[:max_steps]
    return dirs


def angles_to_unitvec(yaw_pitch: torch.Tensor) -> torch.Tensor:
    """
    yaw_pitch: (N,2) [yaw, pitch] radians.
    """
    yaw = yaw_pitch[:, 0]
    pitch = yaw_pitch[:, 1]
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    v = torch.stack([x, y, z], dim=1)
    return F.normalize(v, dim=1)


def mean_angular_error_deg(pred_angles: torch.Tensor,
                           gt_angles: torch.Tensor) -> float:
    """
    Mean angular error (deg) between predicted and GT gaze angles.
    """
    with torch.no_grad():
        v_pred = angles_to_unitvec(pred_angles)
        v_gt = angles_to_unitvec(gt_angles)
        dot = (v_pred * v_gt).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.acos(dot) * 180.0 / math.pi
        return ang.mean().item()


# -------------------- 2. Sequence dataset using basler_eyes_npy --------------------

class EVEEyeSequenceDataset(Dataset):
    """
    Sequence dataset for EyeNet-GRU.

    Uses:
      basler_eyes_npy/*.npy + basler.h5

    Each item:
      imgs   : (T, 1, H, W)  grayscale eye frames
      gazes  : (T, 2)        yaw,pitch (radians)
      pupils : (T, 1)        pupil size (mm)
    """

    def __init__(self, root, split="train",
                 camera="basler",
                 which_eye="left",
                 img_size=(64, 64),
                 seq_len=30,         # T: sequence length
                 step_stride=30,     # how far to move between sequences
                 max_steps=None):
        assert which_eye in ("left", "right")
        self.root = root
        self.split = split
        self.camera = camera
        self.which_eye = which_eye
        self.img_size = img_size
        self.seq_len = seq_len
        self.step_stride = step_stride

        # list of (npy_dir, idxs_array, gaze_seq, pupil_seq)
        self.sequences = []

        step_dirs = list_step_dirs(root, split, max_steps=max_steps)
        for step_dir in step_dirs:
            h5_path = os.path.join(step_dir, f"{camera}.h5")
            npy_dir = os.path.join(step_dir, f"{camera}_eyes_npy")
            if not (os.path.isfile(h5_path) and os.path.isdir(npy_dir)):
                continue

            with h5py.File(h5_path, "r") as f:
                g_key = f"{which_eye}_g_tobii"
                p_key = f"{which_eye}_p"
                if g_key not in f or p_key not in f:
                    continue
                gaze = np.array(f[g_key]["data"])       # (N,2)
                g_valid = np.array(f[g_key]["validity"])
                pupil = np.array(f[p_key]["data"])      # (N,)
                p_valid = np.array(f[p_key]["validity"])

            valid = g_valid.astype(bool) & p_valid.astype(bool)
            N = gaze.shape[0]

            start = 0
            while start + seq_len <= N:
                idxs = np.arange(start, start + seq_len)
                if valid[idxs].all():
                    self.sequences.append(
                        (npy_dir,
                         idxs.astype(int),
                         gaze[idxs].astype(np.float32),
                         pupil[idxs].astype(np.float32))
                    )
                start += self.step_stride

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        npy_dir, idxs, gaze_seq, pupil_seq = self.sequences[idx]

        imgs = []
        for n in idxs:
            fname = f"{int(n):06d}.npy"
            gray = np.load(os.path.join(npy_dir, fname))  # (H0,W0), uint8
            frame_resized = cv2.resize(
                gray, self.img_size, interpolation=cv2.INTER_AREA
            )
            img = frame_resized.astype(np.float32) / 255.0
            imgs.append(np.expand_dims(img, axis=0))      # (1,H,W)

        imgs = np.stack(imgs, axis=0)                    # (T,1,H,W)
        pupils = pupil_seq[:, None]                      # (T,1)

        imgs_t = torch.from_numpy(imgs)                  # (T,1,H,W)
        gazes_t = torch.from_numpy(gaze_seq)             # (T,2)
        pupils_t = torch.from_numpy(pupils)              # (T,1)

        return imgs_t, gazes_t, pupils_t


# -------------------- 3. EyeNet-GRU model --------------------

class EyeNetGRU(nn.Module):
    """
    EyeNet-GRU with MobileNetV3-Small backbone.
    """

    def __init__(self, out_gaze_dim=2, hidden_dim=256):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[0].in_features  # ~576

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.gaze_head = nn.Linear(hidden_dim, out_gaze_dim)
        self.pupil_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (B,T,1,H,W)
        returns:
          gaze:  (B,T,2)
          pupil: (B,T,1)
        """
        B, T, C, H, W = x.shape

        # CNN per frame
        x = x.view(B * T, C, H, W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        f = self.features(x)
        f = self.pool(f).flatten(1)        # (B*T, feat_dim)
        f = f.view(B, T, -1)               # (B,T,feat_dim)

        out, _ = self.gru(f)               # (B,T,hidden_dim)

        gaze = self.gaze_head(out)         # (B,T,2)
        pupil = self.pupil_head(out)       # (B,T,1)
        return gaze, pupil


# -------------------- 4. Train / Eval --------------------

def train_one_epoch_gru(model, loader, optimizer, device):
    model.train()
    running_loss = running_gaze = running_pupil = 0.0

    for imgs, gazes, pupils in loader:
        imgs = imgs.to(device)       # (B,T,1,H,W)
        gazes = gazes.to(device)     # (B,T,2)
        pupils = pupils.to(device)   # (B,T,1)

        pred_gaze, pred_pupil = model(imgs)
        gaze_loss = F.mse_loss(pred_gaze, gazes)
        pupil_loss = F.l1_loss(pred_pupil, pupils)
        loss = gaze_loss + pupil_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_gaze += gaze_loss.item()
        running_pupil += pupil_loss.item()

    n = len(loader)
    return running_loss / n, running_gaze / n, running_pupil / n


@torch.no_grad()
def evaluate_gru(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = total_pupil = 0.0
    all_pred, all_gt = [], []

    for imgs, gazes, pupils in loader:
        imgs = imgs.to(device)
        gazes = gazes.to(device)
        pupils = pupils.to(device)

        pred_gaze, pred_pupil = model(imgs)
        gaze_loss = F.mse_loss(pred_gaze, gazes)
        pupil_loss = F.l1_loss(pred_pupil, pupils)
        loss = gaze_loss + pupil_loss

        total_loss += loss.item()
        total_pupil += pupil_loss.item()
        all_pred.append(pred_gaze.reshape(-1, 2).cpu())
        all_gt.append(gazes.reshape(-1, 2).cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    mae_deg = mean_angular_error_deg(all_pred, all_gt)

    n = len(loader)
    return total_loss / n, total_pupil / n, mae_deg


# -------------------- 5. Main --------------------

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"  # adjust path

    train_dataset = EVEEyeSequenceDataset(
        root=root,
        split="train",
        camera="basler",
        which_eye="left",
        img_size=(64, 64),
        seq_len=30,
        step_stride=30,
        max_steps=None          # or small int while debugging
    )
    val_dataset = EVEEyeSequenceDataset(
        root=root,
        split="val",
        camera="basler",
        which_eye="left",
        img_size=(64, 64),
        seq_len=30,
        step_stride=30,
        max_steps=None
    )

    print("Train sequences:", len(train_dataset))
    print("Val sequences:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = EyeNetGRU(out_gaze_dim=2, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    num_epochs = 5  # increase for better performance
    for epoch in range(num_epochs):
        tr_loss, tr_gaze, tr_pupil = train_one_epoch_gru(model, train_loader, optimizer, device)
        val_loss, val_pupil, val_mae = evaluate_gru(model, val_loader, device)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict()},
            f"checkpoints/eyenet_gru_epoch{epoch+1}.pth",
        )
        print(
            f"[EyeNet-GRU] Epoch {epoch+1}/{num_epochs} "
            f"train_loss={tr_loss:.4f} gazeMSE={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
            f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
            f"val_mean_angular_error={val_mae:.2f} deg"
        )


if __name__ == "__main__":
    main()
