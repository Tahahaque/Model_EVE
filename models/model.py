import os
import glob
import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
def list_step_dirs(root, split="train"):
    """
    root: path to eve_dataset_2 (folder that contains train01, train02, ...)
    split: 'train', 'val', or 'test'
    returns: list of absolute paths to stepXXX_* folders
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
class EVEEyeStaticDataset(Dataset):
    def __init__(self, root, split="train",
                 camera="webcam_c",
                 which_eye="left",
                 img_size=(64, 64),
                 transform=None):
        """
        root: path to eve_dataset_2
        split: 'train' | 'val' | 'test'
        camera: 'basler', 'webcam_l', 'webcam_c', 'webcam_r'
        which_eye: 'left' or 'right'
        img_size: resize target for EyeNet input
        """
        assert which_eye in ("left", "right")
        self.root = root
        self.split = split
        self.camera = camera
        self.which_eye = which_eye
        self.img_size = img_size
        self.transform = transform

        self.samples = []  # list of (mp4_path, frame_idx, gaze_vec(2), pupil_mm)

        step_dirs = list_step_dirs(root, split)
        for step_dir in step_dirs:
            h5_path = os.path.join(step_dir, f"{camera}.h5")
            mp4_path = os.path.join(step_dir, f"{camera}_eyes.mp4")
            if not (os.path.isfile(h5_path) and os.path.isfile(mp4_path)):
                continue

            with h5py.File(h5_path, "r") as f:
                g_key = f"{which_eye}_g_tobii"      # shape (N, 2)
                p_key = f"{which_eye}_p"            # shape (N,)
                if g_key not in f or p_key not in f:
                    continue
                gaze = np.array(f[g_key]["data"])   # EVE stores under ['data']
                pupil = np.array(f[p_key]["data"])

            num_frames = gaze.shape[0]
            for n in range(num_frames):
                self.samples.append(
                    (mp4_path, n, gaze[n].astype(np.float32), float(pupil[n]))
                )

        # cache one VideoCapture per file to avoid reopening too often
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
        # random access: set frame position then read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {mp4_path}")

        # frame is BGR, two-eye image; normalize to single-eye patch
        # Simple starting point: convert to grayscale and resize whole frame.
        # Later you can split left/right halves if needed.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, self.img_size,
                                   interpolation=cv2.INTER_AREA)

        img = frame_resized.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W) for CNN

        if self.transform is not None:
            img = self.transform(img)

        img_t = torch.from_numpy(img)                   # (1,H,W)
        gaze_t = torch.from_numpy(gaze_ang)             # (2,)
        pupil_t = torch.tensor([pupil_mm], dtype=torch.float32)  # (1,)

        return img_t, gaze_t, pupil_t

# if __name__ == "__main__":
#     # adjust this path to your actual dataset root
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     from torch.utils.data import DataLoader
#     import torch

#     dataset = EVEEyeStaticDataset(
#         root=root,
#         split="train",
#         camera="webcam_c",   # or "basler" depending on what you have
#         which_eye="left",
#         img_size=(64, 64),
#     )
#     print("Dataset size:", len(dataset))

#     if len(dataset) > 0:
#         img, gaze, pupil = dataset[0]
#         print("Sample shapes:", img.shape, gaze.shape, pupil.shape)
#     else:
#         print("No samples found – check paths/camera names.")
class EyeNetStatic(nn.Module):
    """
    Static EyeNet: takes a single eye image and predicts gaze angles + pupil size.
    For now we predict 2D angles (theta, phi) directly with an MLP head.
    """
    def __init__(self, out_gaze_dim=2):
        super().__init__()
        # ResNet-18 backbone without final fc
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        feat_dim = 512

        self.gaze_head = nn.Linear(feat_dim, out_gaze_dim)  # 2D angles
        self.pupil_head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        # x: (B, 1, H, W) → replicate to 3 channels for ResNet
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.backbone(x)          # (B, 512)
        gaze = self.gaze_head(feat)      # (B, 2)
        pupil = self.pupil_head(feat)    # (B, 1)
        return gaze, pupil

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

    # 1) Dataset and dataloader
    dataset = EVEEyeStaticDataset(
        root=root,
        split="train",
        camera="webcam_c",   # or "basler"
        which_eye="left",
        img_size=(64, 64),
    )
    print("Dataset size:", len(dataset))

    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # 2) Device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeNetStatic(out_gaze_dim=2).to(device)  # 2D angles

    # 3) Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # 4) Loss function
    def loss_fn(pred_gaze, gt_gaze, pred_pupil, gt_pupil):
        # simple MSE on angles + L1 on pupil
        gaze_loss = F.mse_loss(pred_gaze, gt_gaze)
        pupil_loss = F.l1_loss(pred_pupil, gt_pupil)
        return gaze_loss + pupil_loss, gaze_loss.item(), pupil_loss.item()

    # 5) Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running, running_gaze, running_pupil = 0.0, 0.0, 0.0

        for i, (img, gaze, pupil) in enumerate(loader):
            img = img.to(device)          # (B,1,64,64)
            gaze = gaze.to(device)        # (B,2)
            pupil = pupil.to(device)      # (B,1)

            pred_gaze, pred_pupil = model(img)
            loss, lg, lp = loss_fn(pred_gaze, gaze, pred_pupil, pupil)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            running_gaze += lg
            running_pupil += lp

            if (i + 1) % 50 == 0:
                n = i + 1
                print(
                    f"Epoch {epoch+1} Iter {n}/{len(loader)} "
                    f"loss={running/n:.4f} gaze={running_gaze/n:.4f} pupil={running_pupil/n:.4f}"
                )

        print(f"Epoch {epoch+1} finished, mean loss={running/len(loader):.4f}")
    for idx in range(4):
        img_t, gaze_t, pupil_t = dataset[idx]   # img_t: (1,64,64)
        img = img_t.numpy().squeeze(0)          # (64,64), values in [0,1]

        plt.figure()
        plt.title(f"idx={idx}, gaze={gaze_t.numpy()}, pupil={pupil_t.item():.2f} mm")
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.show()


