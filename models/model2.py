import os, h5py, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

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
    return np.stack(frames)  # (N, H, W, 3)

class EyeNetStepDataset(Dataset):
    def __init__(self, step_dir):
        self.step_dir = step_dir
        self.eye_frames = load_video(os.path.join(step_dir, "basler_eyes.mp4"))

        with h5py.File(os.path.join(step_dir, "basler.h5"), "r") as f:
            g = f["left_g_tobii"]["data"][()]      # (N, 2)
            g_valid = f["left_g_tobii"]["validity"][()]
            p = f["left_p"]["data"][()]            # (N,)
            p_valid = f["left_p"]["validity"][()]

        valid = g_valid.astype(bool) & p_valid.astype(bool)
        self.indices = np.where(valid)[0]
        self.gaze = g
        self.pupil = p

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        img = self.eye_frames[i]
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (3,128,128)

        gaze = self.gaze[i].astype(np.float32)     # (2,)
        pupil = np.array([self.pupil[i]], np.float32)  # (1,)

        return torch.from_numpy(img), torch.from_numpy(gaze), torch.from_numpy(pupil)


step_dir = "eve_dataset_2/train01/step007_image_MIT-i2277207572"  # adjust

eye_ds = EyeNetStepDataset(step_dir)
print("EyeNet samples:", len(eye_ds))

eye_patch, gaze_dir, pupil = eye_ds[0]
print("Eye patch shape:", eye_patch.shape)   # -> torch.Size([3,128,128])
print("Gaze dir:", gaze_dir)                 # e.g. tensor([yaw, pitch])
print("Pupil size:", pupil)                  # e.g. tensor([3.2])
from torch.utils.data import DataLoader
loader = DataLoader(eye_ds, batch_size=4, shuffle=True)
imgs, gazes, pupils = next(iter(loader))
print(imgs.shape)    # torch.Size([4, 3, 128, 128])
print(gazes.shape)   # torch.Size([4, 2])
print(pupils.shape)  # torch.Size([4, 1])


# def load_timestamps(path):
#     return np.loadtxt(path)

# def make_gaussian_heatmap(center_xy, H=72, W=128, sigma=3.0):
#     x0, y0 = center_xy
#     xs = np.arange(W)
#     ys = np.arange(H)
#     X, Y = np.meshgrid(xs, ys)
#     heat = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
#     heat /= (heat.max() + 1e-8)
#     return heat[None, ...].astype(np.float32)  # (1,H,W)


# class GazeRefineNetStepDataset(Dataset):
#     def __init__(self, step_dir):
#         self.step_dir = step_dir

#         # screen frames + timestamps
#         self.screen_frames = load_video(os.path.join(step_dir, "screen.128x72.mp4"))
#         self.screen_ts = load_timestamps(os.path.join(step_dir, "screen.timestamps.txt"))

#         # PoG labels from HDF5 (full screen pixels 1920x1080)
#         with h5py.File(os.path.join(step_dir, "basler.h5"), "r") as f:
#             l = f["left_PoG_tobii"]["data"][()]
#             r = f["right_PoG_tobii"]["data"][()]
#             lv = f["left_PoG_tobii"]["validity"][()]
#             rv = f["right_PoG_tobii"]["validity"][()]
#             self.label_ts = f["timestamps"][()]   # or align via basler.timestamps.txt

#         valid = lv.astype(bool) & rv.astype(bool)
#         pog_full = 0.5 * (l + r)      # (N,2), full-res pixels
#         self.pog = pog_full[valid]
#         self.label_ts = self.label_ts[valid]

#         # downscale full-res PoG to 128x72 coords for fake init
#         screen_w, screen_h = 1920.0, 1080.0
#         scale_x = 128.0 / screen_w
#         scale_y = 72.0 / screen_h
#         self.init_pog_pixels = np.stack([
#             self.pog[:, 0] * scale_x,
#             self.pog[:, 1] * scale_y
#         ], axis=-1).astype(np.float32)

#         # pair each label with closest screen frame
#         self.pairs = []
#         for i, t in enumerate(self.label_ts):
#             j = int(np.argmin(np.abs(self.screen_ts - t)))
#             if 0 <= j < len(self.screen_frames):
#                 self.pairs.append((i, j))

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         li, sj = self.pairs[idx]

#         scr = self.screen_frames[sj].astype(np.float32) / 255.0
#         scr = scr.transpose(2, 0, 1)  # (3,72,128)

#         init_xy = self.init_pog_pixels[li]           # (2,)
#         init_heat = make_gaussian_heatmap(init_xy)   # (1,72,128)

#         target_PoG = self.pog[li].astype(np.float32) # (2,), full-res pixels

#         return torch.from_numpy(scr), torch.from_numpy(init_heat), torch.from_numpy(target_PoG)


# ref_ds = GazeRefineNetStepDataset(step_dir)
# print("RefineNet samples:", len(ref_ds))

# screen_frame, init_heatmap, target_PoG = ref_ds[0]
# print("Screen frame shape:", screen_frame.shape)   # -> torch.Size([3,72,128])
# print("Init heatmap shape:", init_heatmap.shape)   # -> torch.Size([1,72,128])
# print("Target PoG:", target_PoG)                   # tensor([x_px, y_px])
