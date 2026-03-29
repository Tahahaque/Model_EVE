import os
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def load_timestamps(path):
    return np.loadtxt(path)  # shape (N,), in seconds

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    cap.release()
    return np.stack(frames)  # (T, H, W, 3)

def closest_index(ts_array, t):
    return int(np.argmin(np.abs(ts_array - t)))

class EyeNetStepDataset(Dataset):
    """
    One stepXXX folder → training samples for EyeNet.
    Uses: basler_eyes.mp4 + basler.h5
    Outputs per index:
        eye_patch:  (3, 128, 128) float32 in [0,1]
        gaze_dir:   (2,) yaw, pitch (from left_g_tobii)
        pupil:      (1,) pupil size in mm (left_p)
    """

    def __init__(self, step_dir, use_right_eye=False, transform=None):
        self.step_dir = step_dir
        self.use_right = use_right_eye
        self.transform = transform

        # load video
        eyes_path = os.path.join(step_dir, "basler_eyes.mp4")
        self.eye_frames = load_video(eyes_path)     # (N, H, W, 3)

        # load labels
        h5_path = os.path.join(step_dir, "basler.h5")
        with h5py.File(h5_path, "r") as f:
            if not use_right_eye:
                g = f["left_g_tobii"]["data"][()]      # (N, 2)
                g_valid = f["left_g_tobii"]["validity"][()]
                p = f["left_p"]["data"][()]            # (N,)
                p_valid = f["left_p"]["validity"][()]
            else:
                g = f["right_g_tobii"]["data"][()]
                g_valid = f["right_g_tobii"]["validity"][()]
                p = f["right_p"]["data"][()]
                p_valid = f["right_p"]["validity"][()]

        valid = (g_valid.astype(bool) & p_valid.astype(bool))
        self.indices = np.where(valid)[0]
        self.gaze = g
        self.pupil = p

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        img = self.eye_frames[i]  # (H, W, 3)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (3, 128, 128)

        gaze = self.gaze[i].astype(np.float32)   # (2,)
        pupil = np.array([self.pupil[i]], np.float32)

        if self.transform is not None:
            img = self.transform(img)

        return torch.from_numpy(img), torch.from_numpy(gaze), torch.from_numpy(pupil)

def make_gaussian_heatmap(center_xy, H=72, W=128, sigma=3.0):
    """
    center_xy: (2,) in pixel coordinates of downscaled screen (W, H)
    returns (1, H, W)
    """
    x0, y0 = center_xy
    xs = np.arange(W)
    ys = np.arange(H)
    X, Y = np.meshgrid(xs, ys)
    heat = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    heat = heat / (heat.max() + 1e-8)
    return heat[None, ...].astype(np.float32)

class GazeRefineNetStepDataset(Dataset):
    """
    One stepXXX folder → training samples for GazeRefineNet.
    Uses: screen.128x72.mp4 + basler.h5 + initial PoG from EyeNet.
    Outputs per index:
        screen_frame:   (3, 72, 128)
        init_heatmap:   (1, 72, 128)
        target_PoG:     (2,) in cm or normalized pixels
    """

    def __init__(self, step_dir, init_pog_pixels, use_face_pog=False):
        self.step_dir = step_dir
        self.screen_frames = load_video(os.path.join(step_dir, "screen.128x72.mp4"))  # (Ns, 72, 128, 3)
        self.screen_ts = load_timestamps(os.path.join(step_dir, "screen.timestamps.txt"))

        # load PoG labels (Tobii)
        h5_path = os.path.join(step_dir, "basler.h5")
        with h5py.File(h5_path, "r") as f:
            if use_face_pog:
                pog = f["face_PoG_tobii"]["data"][()]   # (Ng, 2) in full-screen pixels
                pog_valid = f["face_PoG_tobii"]["validity"][()]
            else:
                # example: average left/right PoG in screen pixels
                l = f["left_PoG_tobii"]["data"][()]
                r = f["right_PoG_tobii"]["data"][()]
                lv = f["left_PoG_tobii"]["validity"][()]
                rv = f["right_PoG_tobii"]["validity"][()]
                pog = 0.5 * (l + r)
                pog_valid = (lv.astype(bool) & rv.astype(bool))

            # timestamps for basler / labels (stored somewhere in HDF; you might
            # also read basler.timestamps.txt and align separately)
            # For pseudo code we assume an array:
            label_ts = f["timestamps"][()]  # shape (Ng,)

        self.label_ts = label_ts
        self.pog = pog
        self.pog_valid = pog_valid.astype(bool)

        # initial PoG predictions from EyeNet, in **downscaled** screen pixels
        # init_pog_pixels: np.array (Ng, 2) already aligned with label_ts
        self.init_pog_pixels = init_pog_pixels.astype(np.float32)

        # choose valid indices where we have label, init prediction, and a screen frame
        self.indices = []
        for i in range(len(self.label_ts)):
            if not self.pog_valid[i]:
                continue
            t = self.label_ts[i]
            j = closest_index(self.screen_ts, t)
            if 0 <= j < len(self.screen_frames):
                self.indices.append((i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        li, sj = self.indices[idx]

        # screen frame
        scr = self.screen_frames[sj]  # (72, 128, 3)
        scr = scr.astype(np.float32) / 255.0
        scr = scr.transpose(2, 0, 1)  # (3, 72, 128)

        # initial PoG heatmap (downscaled coordinates)
        init_xy = self.init_pog_pixels[li]  # (2,) in 128x72 coords
        init_heat = make_gaussian_heatmap(init_xy, H=72, W=128)  # (1,72,128)

        # target PoG (you can keep pixels or convert to cm using mm_per_pixel)
        target_xy = self.pog[li].astype(np.float32)  # full-res pixels; convert if needed

        return (
            torch.from_numpy(scr),
            torch.from_numpy(init_heat),
            torch.from_numpy(target_xy),
        )
