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
import matplotlib.pyplot as plt

# -------------------- Helpers --------------------

def list_step_dirs(root: str, split: str = "train"):
    """
    root: path to eve_dataset_2
    split: 'train' | 'val' | 'test'
    Returns list of stepXXX_* dirs under train*, val*, test*.
    """
    if split == "train":
        pat = os.path.join(root, "train*", "step*")
    elif split == "val":
        pat = os.path.join(root, "val*", "step*")
    elif split == "test":
        pat = os.path.join(root, "test*", "step*")
    else:
        raise ValueError(split)
    return sorted(glob.glob(pat))


def angles_to_unitvec(yaw_pitch: torch.Tensor) -> torch.Tensor:
    """
    yaw_pitch: (B,2) [yaw, pitch] in radians.
    Returns: (B,3) unit vectors.
    """
    yaw = yaw_pitch[:, 0]
    pitch = yaw_pitch[:, 1]
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    v = torch.stack([x, y, z], dim=1)
    v = F.normalize(v, dim=1)
    return v


def mean_angular_error_deg(pred_angles: torch.Tensor,
                           gt_angles: torch.Tensor) -> float:
    """
    pred_angles, gt_angles: (N,2), radians.
    Returns mean angular error in degrees.
    """
    with torch.no_grad():
        v_pred = angles_to_unitvec(pred_angles)
        v_gt = angles_to_unitvec(gt_angles)
        dot = (v_pred * v_gt).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.acos(dot) * 180.0 / math.pi
        return ang.mean().item()


# -------------------- Dataset --------------------

class EVEEyeStaticDataset(Dataset):
    """
    Single-frame EyeNet dataset using <camera>_eyes.mp4 + <camera>.h5
    Labels: {which_eye}_g_tobii (yaw,pitch) and {which_eye}_p (pupil size).
    """

    def __init__(self, root, split="train",
                 camera="webcam_c",
                 which_eye="left",
                 img_size=(64, 64)):
        assert which_eye in ("left", "right")
        self.root = root
        self.split = split
        self.camera = camera
        self.which_eye = which_eye
        self.img_size = img_size

        self.samples = []  # list of (mp4_path, frame_idx, gaze_vec(2), pupil_mm)

        step_dirs = list_step_dirs(root, split)
        for step_dir in step_dirs:
            h5_path = os.path.join(step_dir, f"{camera}.h5")
            mp4_path = os.path.join(step_dir, f"{camera}_eyes.mp4")
            if not (os.path.isfile(h5_path) and os.path.isfile(mp4_path)):
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
            idxs = np.where(valid)[0]

            for n in idxs:
                self.samples.append(
                    (mp4_path,
                     int(n),
                     gaze[n].astype(np.float32),
                     float(pupil[n]))
                )

        self._caps = {}

    def __len__(self):
        return len(self.samples)

    def _get_cap(self, mp4_path):
        cap = self._caps.get(mp4_path, None)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(mp4_path)
            self._caps[mp4_path] = cap
        return cap

    def __getitem__(self, idx):
        mp4_path, frame_idx, gaze_ang, pupil_mm = self.samples[idx]

        cap = self._get_cap(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {mp4_path}")

        # BGR → gray, resize to img_size
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, self.img_size,
                                   interpolation=cv2.INTER_AREA)

        img = frame_resized.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1,H,W)

        img_t = torch.from_numpy(img)                      # (1,H,W)
        gaze_t = torch.from_numpy(gaze_ang)                # (2,)
        pupil_t = torch.tensor([pupil_mm], dtype=torch.float32)  # (1,)

        return img_t, gaze_t, pupil_t


# -------------------- Model (MobileNetV3-Small) --------------------

class EyeNetStatic(nn.Module):
    """
    Lightweight EyeNet: MobileNetV3-Small backbone.
    Single eye image → gaze angles (yaw,pitch) + pupil size.
    """

    def __init__(self, out_gaze_dim=2):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features          # conv trunk
        self.pool = nn.AdaptiveAvgPool2d(1)       # global avg pool
        feat_dim = backbone.classifier[0].in_features  # usually 576

        self.gaze_head = nn.Linear(feat_dim, out_gaze_dim)
        self.pupil_head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        # x: (B,1,H,W) → (B,3,H,W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        f = self.features(x)          # (B,C,H',W')
        f = self.pool(f).flatten(1)   # (B,feat_dim)
        gaze = self.gaze_head(f)      # (B,2)
        pupil = self.pupil_head(f)    # (B,1)
        return gaze, pupil


# -------------------- Train / Eval --------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss, running_gaze, running_pupil = 0.0, 0.0, 0.0

    for img, gaze, pupil in loader:
        img = img.to(device)
        gaze = gaze.to(device)
        pupil = pupil.to(device)

        pred_gaze, pred_pupil = model(img)
        gaze_loss = F.mse_loss(pred_gaze, gaze)
        pupil_loss = F.l1_loss(pred_pupil, pupil)
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
def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    total_loss, total_pupil = 0.0, 0.0
    all_pred, all_gt = [], []

    for img, gaze, pupil in loader:
        img = img.to(device)
        gaze = gaze.to(device)
        pupil = pupil.to(device)

        pred_gaze, pred_pupil = model(img)
        gaze_loss = F.mse_loss(pred_gaze, gaze)
        pupil_loss = F.l1_loss(pred_pupil, pupil)
        loss = gaze_loss + pupil_loss

        total_loss += loss.item()
        total_pupil += pupil_loss.item()
        all_pred.append(pred_gaze.cpu())
        all_gt.append(gaze.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    mae_deg = mean_angular_error_deg(all_pred, all_gt)

    n = len(loader)
    return total_loss / n, total_pupil / n, mae_deg


def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"  # adjust if needed

    # train*: train01, train02, ...
    train_dataset = EVEEyeStaticDataset(
        root=root,
        split="train",
        camera="webcam_c",       # or "basler"
        which_eye="left",
        img_size=(64, 64),
    )

    # val*: val01, ...
    val_dataset = EVEEyeStaticDataset(
        root=root,
        split="val",
        camera="webcam_c",
        which_eye="left",
        img_size=(64, 64),
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=0
    )

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Device:", device)

    model = EyeNetStatic(out_gaze_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    num_epochs = 5  # increase for better performance
    for epoch in range(num_epochs):
        train_loss, train_gaze, train_pupil = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_pupil, val_mae_deg = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"train_loss={train_loss:.4f} gazeMSE={train_gaze:.4f} pupilL1={train_pupil:.4f} | "
            f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
            f"val_mean_angular_error={val_mae_deg:.2f} deg"
        )


if __name__ == "__main__":
    main()
