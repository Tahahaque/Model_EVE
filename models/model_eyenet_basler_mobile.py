# import os
# import glob
# import h5py
# import cv2
# import numpy as np
# import math
# from typing import Tuple

# import torch
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models


# # ---------- Helpers ----------

# def list_step_dirs(root: str, split: str = "train", max_steps=None):
#     if split == "train":
#         pat = os.path.join(root, "train*", "step*")
#     elif split == "val":
#         pat = os.path.join(root, "val*", "step*")
#     elif split == "test":
#         pat = os.path.join(root, "test*", "step*")
#     else:
#         raise ValueError(split)
#     dirs = sorted(glob.glob(pat))
#     if max_steps is not None:
#         dirs = dirs[:max_steps]
#     return dirs


# def angles_to_unitvec(yaw_pitch: torch.Tensor) -> torch.Tensor:
#     yaw = yaw_pitch[:, 0]
#     pitch = yaw_pitch[:, 1]
#     x = torch.cos(pitch) * torch.sin(yaw)
#     y = torch.sin(pitch)
#     z = torch.cos(pitch) * torch.cos(yaw)
#     v = torch.stack([x, y, z], dim=1)
#     return F.normalize(v, dim=1)


# def mean_angular_error_deg(pred_angles: torch.Tensor,
#                            gt_angles: torch.Tensor) -> float:
#     with torch.no_grad():
#         v_pred = angles_to_unitvec(pred_angles)
#         v_gt = angles_to_unitvec(gt_angles)
#         dot = (v_pred * v_gt).sum(dim=1).clamp(-1.0, 1.0)
#         ang = torch.acos(dot) * 180.0 / math.pi
#         return ang.mean().item()


# # ---------- Dataset (uses Basler) ----------

# class EVEEyeStaticDataset(Dataset):
#     """
#     Uses basler_eyes.mp4 + basler.h5
#     Labels: {which_eye}_g_tobii (yaw,pitch) and {which_eye}_p (pupil).
#     """

#     def __init__(self, root, split="train",
#                  camera="basler",
#                  which_eye="left",
#                  img_size=(64, 64),
#                  max_steps=None):
#         assert which_eye in ("left", "right")
#         self.root = root
#         self.split = split
#         self.camera = camera
#         self.which_eye = which_eye
#         self.img_size = img_size

#         self.samples = []  # (mp4_path, frame_idx, gaze_vec(2), pupil_mm)
#         step_dirs = list_step_dirs(root, split, max_steps=max_steps)

#         for step_dir in step_dirs:
#             h5_path = os.path.join(step_dir, f"{camera}.h5")
#             mp4_path = os.path.join(step_dir, f"{camera}_eyes.mp4")
#             if not (os.path.isfile(h5_path) and os.path.isfile(mp4_path)):
#                 continue

#             with h5py.File(h5_path, "r") as f:
#                 g_key = f"{which_eye}_g_tobii"
#                 p_key = f"{which_eye}_p"
#                 if g_key not in f or p_key not in f:
#                     continue
#                 gaze = np.array(f[g_key]["data"])       # (N,2)
#                 g_valid = np.array(f[g_key]["validity"])
#                 pupil = np.array(f[p_key]["data"])      # (N,)
#                 p_valid = np.array(f[p_key]["validity"])

#             valid = g_valid.astype(bool) & p_valid.astype(bool)
#             idxs = np.where(valid)[0]

#             for n in idxs:
#                 self.samples.append(
#                     (mp4_path,
#                      int(n),
#                      gaze[n].astype(np.float32),
#                      float(pupil[n]))
#                 )

#         self._caps = {}

#     def __len__(self):
#         return len(self.samples)

#     def _get_cap(self, mp4_path):
#         cap = self._caps.get(mp4_path, None)
#         if cap is None or not cap.isOpened():
#             cap = cv2.VideoCapture(mp4_path)
#             self._caps[mp4_path] = cap
#         return cap

#     def __getitem__(self, idx):
#         mp4_path, frame_idx, gaze_ang, pupil_mm = self.samples[idx]

#         cap = self._get_cap(mp4_path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ok, frame = cap.read()
#         if not ok:
#             raise RuntimeError(f"Failed to read frame {frame_idx} from {mp4_path}")

#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame_resized = cv2.resize(frame_gray, self.img_size,
#                                    interpolation=cv2.INTER_AREA)

#         img = frame_resized.astype(np.float32) / 255.0
#         img = np.expand_dims(img, axis=0)  # (1,H,W)

#         img_t = torch.from_numpy(img)
#         gaze_t = torch.from_numpy(gaze_ang)
#         pupil_t = torch.tensor([pupil_mm], dtype=torch.float32)

#         return img_t, gaze_t, pupil_t


# # ---------- Train / Eval shared ----------

# def train_one_epoch(model, loader, optimizer, device):
#     model.train()
#     running_loss = running_gaze = running_pupil = 0.0

#     for img, gaze, pupil in loader:
#         img = img.to(device)
#         gaze = gaze.to(device)
#         pupil = pupil.to(device)

#         pred_gaze, pred_pupil = model(img)
#         gaze_loss = F.mse_loss(pred_gaze, gaze)
#         pupil_loss = F.l1_loss(pred_pupil, pupil)
#         loss = gaze_loss + pupil_loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         running_gaze += gaze_loss.item()
#         running_pupil += pupil_loss.item()

#     n = len(loader)
#     return running_loss / n, running_gaze / n, running_pupil / n


# @torch.no_grad()
# def evaluate(model, loader, device) -> Tuple[float, float, float]:
#     model.eval()
#     total_loss = total_pupil = 0.0
#     all_pred, all_gt = [], []

#     for img, gaze, pupil in loader:
#         img = img.to(device)
#         gaze = gaze.to(device)
#         pupil = pupil.to(device)

#         pred_gaze, pred_pupil = model(img)
#         gaze_loss = F.mse_loss(pred_gaze, gaze)
#         pupil_loss = F.l1_loss(pred_pupil, pupil)
#         loss = gaze_loss + pupil_loss

#         total_loss += loss.item()
#         total_pupil += pupil_loss.item()
#         all_pred.append(pred_gaze.cpu())
#         all_gt.append(gaze.cpu())

#     all_pred = torch.cat(all_pred, dim=0)
#     all_gt = torch.cat(all_gt, dim=0)
#     mae_deg = mean_angular_error_deg(all_pred, all_gt)

#     n = len(loader)
#     return total_loss / n, total_pupil / n, mae_deg

# class EyeNetMobileNet(nn.Module):
#     """
#     EyeNet with MobileNetV3-Small backbone.
#     """

#     def __init__(self, out_gaze_dim=2):
#         super().__init__()
#         backbone = models.mobilenet_v3_small(weights=None)
#         self.features = backbone.features
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         feat_dim = backbone.classifier[0].in_features  # usually 576

#         self.gaze_head = nn.Linear(feat_dim, out_gaze_dim)
#         self.pupil_head = nn.Linear(feat_dim, 1)

#     def forward(self, x):
#         # (B,1,H,W) -> (B,3,H,W)
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)

#         f = self.features(x)
#         f = self.pool(f).flatten(1)
#         gaze = self.gaze_head(f)
#         pupil = self.pupil_head(f)
#         return gaze, pupil
    
# def main_mobilenet():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     train_dataset = EVEEyeStaticDataset(
#         root=root, split="train", camera="basler", which_eye="left",
#         img_size=(64, 64), max_steps=None
#     )
#     val_dataset = EVEEyeStaticDataset(
#         root=root, split="val", camera="basler", which_eye="left",
#         img_size=(64, 64), max_steps=None
#     )

#     print("Train samples:", len(train_dataset))
#     print("Val samples:", len(val_dataset))

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print("Device:", device)

#     model = EyeNetMobileNet(out_gaze_dim=2).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

#     num_epochs = 3
#     for epoch in range(num_epochs):
#         tr_loss, tr_gaze, tr_pupil = train_one_epoch(model, train_loader, optimizer, device)
#         val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)
#         print(
#             f"[MobileNet] Epoch {epoch+1}/{num_epochs} "
#             f"train_loss={tr_loss:.4f} gazeMSE={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
#             f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
#             f"val_mean_angular_error={val_mae:.2f} deg"
#         )
# class EyeNetEfficientNet(nn.Module):
#     """
#     EyeNet with EfficientNet-B0 backbone.
#     """

#     def __init__(self, out_gaze_dim=2):
#         super().__init__()
#         backbone = models.efficientnet_b0(weights=None)
#         self.features = backbone.features
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         feat_dim = backbone.classifier[1].in_features  # final linear in orig model

#         self.gaze_head = nn.Linear(feat_dim, out_gaze_dim)
#         self.pupil_head = nn.Linear(feat_dim, 1)

#     def forward(self, x):
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)

#         f = self.features(x)
#         f = self.pool(f).flatten(1)
#         gaze = self.gaze_head(f)
#         pupil = self.pupil_head(f)
#         return gaze, pupil
# def main_efficientnet():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     train_dataset = EVEEyeStaticDataset(
#         root=root, split="train", camera="basler", which_eye="left",
#         img_size=(64, 64), max_steps=None
#     )
#     val_dataset = EVEEyeStaticDataset(
#         root=root, split="val", camera="basler", which_eye="left",
#         img_size=(64, 64), max_steps=None
#     )

#     print("Train samples:", len(train_dataset))
#     print("Val samples:", len(val_dataset))

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print("Device:", device)

#     model = EyeNetEfficientNet(out_gaze_dim=2).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

#     num_epochs = 3
#     for epoch in range(num_epochs):
#         tr_loss, tr_gaze, tr_pupil = train_one_epoch(model, train_loader, optimizer, device)
#         val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)
#         print(
#             f"[EfficientNet] Epoch {epoch+1}/{num_epochs} "
#             f"train_loss={tr_loss:.4f} gazeMSE={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
#             f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
#             f"val_mean_angular_error={val_mae:.2f} deg"
#         )

# if __name__ == "__main__":
#     main_mobilenet()
#     # main_efficientnet()

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


# ---------- Helpers ----------

def list_step_dirs(root: str, split: str = "train", max_steps=None):
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
    yaw = yaw_pitch[:, 0]
    pitch = yaw_pitch[:, 1]
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    v = torch.stack([x, y, z], dim=1)
    return F.normalize(v, dim=1)


def mean_angular_error_deg(pred_angles: torch.Tensor,
                           gt_angles: torch.Tensor) -> float:
    with torch.no_grad():
        v_pred = angles_to_unitvec(pred_angles)
        v_gt = angles_to_unitvec(gt_angles)
        dot = (v_pred * v_gt).sum(dim=1).clamp(-1.0, 1.0)
        ang = torch.acos(dot) * 180.0 / math.pi
        return ang.mean().item()


# ---------- Dataset using basler_eyes_npy ----------

class EVEEyeStaticDataset(Dataset):
    """
    Uses basler_eyes_npy/*.npy + basler.h5
    Labels: {which_eye}_g_tobii (yaw,pitch) and {which_eye}_p (pupil).
    """

    def __init__(self, root, split="train",
                 camera="basler",
                 which_eye="left",
                 img_size=(64, 64),
                 max_steps=None):
        assert which_eye in ("left", "right")
        self.root = root
        self.split = split
        self.camera = camera
        self.which_eye = which_eye
        self.img_size = img_size

        # (npy_dir, frame_name, gaze_vec(2), pupil_mm)
        self.samples = []
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
            idxs = np.where(valid)[0]

            for n in idxs:
                fname = f"{int(n):06d}.npy"
                self.samples.append(
                    (npy_dir,
                     fname,
                     gaze[n].astype(np.float32),
                     float(pupil[n]))
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_dir, fname, gaze_ang, pupil_mm = self.samples[idx]

        path = os.path.join(npy_dir, fname)
        gray = np.load(path)  # (H_orig, W_orig), uint8
        frame_resized = cv2.resize(gray, self.img_size,
                                   interpolation=cv2.INTER_AREA)

        img = frame_resized.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1,H,W)

        img_t = torch.from_numpy(img)
        gaze_t = torch.from_numpy(gaze_ang)
        pupil_t = torch.tensor([pupil_mm], dtype=torch.float32)

        return img_t, gaze_t, pupil_t


# ---------- Train / Eval shared ----------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = running_gaze = running_pupil = 0.0

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
    total_loss = total_pupil = 0.0
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


# ---------- Models ----------

class EyeNetMobileNet(nn.Module):
    def __init__(self, out_gaze_dim=2):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[0].in_features

        self.gaze_head = nn.Linear(feat_dim, out_gaze_dim)
        self.pupil_head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        f = self.features(x)
        f = self.pool(f).flatten(1)
        gaze = self.gaze_head(f)
        pupil = self.pupil_head(f)
        return gaze, pupil


class EyeNetEfficientNet(nn.Module):
    def __init__(self, out_gaze_dim=2):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[1].in_features

        self.gaze_head = nn.Linear(feat_dim, out_gaze_dim)
        self.pupil_head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        f = self.features(x)
        f = self.pool(f).flatten(1)
        gaze = self.gaze_head(f)
        pupil = self.pupil_head(f)
        return gaze, pupil


# ---------- Main ----------

def main_mobilenet():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

    train_dataset = EVEEyeStaticDataset(
        root=root, split="train", camera="basler", which_eye="left",
        img_size=(64, 64), max_steps=None
    )
    val_dataset = EVEEyeStaticDataset(
        root=root, split="val", camera="basler", which_eye="left",
        img_size=(64, 64), max_steps=None
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = EyeNetMobileNet(out_gaze_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        tr_loss, tr_gaze, tr_pupil = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)
        print(
            f"[MobileNet] Epoch {epoch+1}/{num_epochs} "
            f"train_loss={tr_loss:.4f} gazeMSE={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
            f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
            f"val_mean_angular_error={val_mae:.2f} deg"
        )


def main_efficientnet():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

    train_dataset = EVEEyeStaticDataset(
        root=root, split="train", camera="basler", which_eye="left",
        img_size=(64, 64), max_steps=None
    )
    val_dataset = EVEEyeStaticDataset(
        root=root, split="val", camera="basler", which_eye="left",
        img_size=(64, 64), max_steps=None
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = EyeNetEfficientNet(out_gaze_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        tr_loss, tr_gaze, tr_pupil = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)
        print(
            f"[EfficientNet] Epoch {epoch+1}/{num_epochs} "
            f"train_loss={tr_loss:.4f} gazeMSE={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
            f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
            f"val_mean_angular_error={val_mae:.2f} deg"
        )


if __name__ == "__main__":
    main_mobilenet()
    # main_efficientnet()

