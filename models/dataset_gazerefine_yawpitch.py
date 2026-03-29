# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import cv2

# class GazeRefineYawPitchDataset(Dataset):
#     """
#     Uses:
#       screen.128x72.mp4
#       eyenet_yawpitch.npy
#       gt_yawpitch.npy

#     Returns per item:
#       screen_seq: (T,3,72,128)
#       eyenet_seq: (T,2)
#       gt_seq    : (T,2)
#     """
#     def __init__(self, root, split="train",
#                  seq_len=30,
#                  step_stride=30,
#                  max_steps=None):
#         self.root = root
#         self.split = split
#         self.seq_len = seq_len
#         self.step_stride = step_stride

#         if split.startswith("train"):
#             pat = os.path.join(root, "train*", "step*")
#         elif split.startswith("val"):
#             pat = os.path.join(root, "val*", "step*")
#         elif split.startswith("test"):
#             pat = os.path.join(root, "test*", "step*")
#         else:
#             raise ValueError(split)

#         self.step_dirs = sorted(glob.glob(pat))
#         if max_steps is not None:
#             self.step_dirs = self.step_dirs[:max_steps]

#         self.index = []   # (step_idx, start)
#         for step_i, step_dir in enumerate(self.step_dirs):
#             e_path = os.path.join(step_dir, "eyenet_yawpitch.npy")
#             g_path = os.path.join(step_dir, "gt_yawpitch.npy")
#             if not (os.path.exists(e_path) and os.path.exists(g_path)):
#                 continue
#             eyenet = np.load(e_path)
#             gt = np.load(g_path)
#             assert eyenet.shape == gt.shape
#             N = eyenet.shape[0]
#             for start in range(0, N - seq_len + 1, step_stride):
#                 self.index.append((step_i, start))

#         self._screen_cache = {}
#         self._eyenet_cache = {}
#         self._gt_cache = {}

#     def __len__(self):
#         return len(self.index)

#     def _load_step(self, step_dir):
#         if step_dir in self._screen_cache:
#             return (self._screen_cache[step_dir],
#                     self._eyenet_cache[step_dir],
#                     self._gt_cache[step_dir])

#         cap = cv2.VideoCapture(os.path.join(step_dir, "screen.128x72.mp4"))
#         frames = []
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = frame.astype(np.float32) / 255.0
#             frame = np.transpose(frame, (2, 0, 1))  # (3,72,128)
#             frames.append(frame)
#         cap.release()
#         screen = np.stack(frames, axis=0)  # (N,3,72,128)

#         eyenet = np.load(os.path.join(step_dir, "eyenet_yawpitch.npy"))  # (N,2)
#         gt = np.load(os.path.join(step_dir, "gt_yawpitch.npy"))          # (N,2)

#         assert screen.shape[0] == eyenet.shape[0] == gt.shape[0]

#         self._screen_cache[step_dir] = screen
#         self._eyenet_cache[step_dir] = eyenet
#         self._gt_cache[step_dir] = gt
#         return screen, eyenet, gt
#     # def _load_step(self, step_dir):
#     #     if step_dir in self._screen_cache:
#     #         return (self._screen_cache[step_dir],
#     #                 self._eyenet_cache[step_dir],
#     #                 self._gt_cache[step_dir])

#     #     cap = cv2.VideoCapture(os.path.join(step_dir, "screen.128x72.mp4"))
#     #     frames = []
#     #     while True:
#     #         ok, frame = cap.read()
#     #         if not ok:
#     #             break
#     #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     #         frame = frame.astype(np.float32) / 255.0
#     #         frame = np.transpose(frame, (2, 0, 1))
#     #         frames.append(frame)
#     #     cap.release()
#     #     screen = np.stack(frames, axis=0)  # (Ns,3,72,128)

#     #     eyenet = np.load(os.path.join(step_dir, "eyenet_yawpitch.npy"))  # (Ne,2)
#     #     gt = np.load(os.path.join(step_dir, "gt_yawpitch.npy"))          # (Ng,2)

#     # # align lengths by truncating to min
#     #     N = min(screen.shape[0], eyenet.shape[0], gt.shape[0])
#     #     screen = screen[:N]
#     #     yenet = eyenet[:N]
#     #     gt = gt[:N]

#     #     self._screen_cache[step_dir] = screen
#     #     self._eyenet_cache[step_dir] = eyenet
#     #     self._gt_cache[step_dir] = gt
#     #     return screen, eyenet, gt

#     def __getitem__(self, idx):
#         step_i, start = self.index[idx]
#         step_dir = self.step_dirs[step_i]

#         screen, eyenet, gt = self._load_step(step_dir)
#         end = start + self.seq_len

#         screen_seq = torch.from_numpy(screen[start:end])  # (T,3,72,128)
#         eyenet_seq = torch.from_numpy(eyenet[start:end])  # (T,2)
#         gt_seq = torch.from_numpy(gt[start:end])          # (T,2)

#         return {
#             "screen_seq": screen_seq,
#             "eyenet_seq": eyenet_seq,
#             "gt_yawpitch_seq": gt_seq,
#         }


#newwwwww !!!!

# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import cv2


# class GazeRefineYawPitchDataset(Dataset):
#     """
#     Uses:
#       screen.128x72.mp4
#       eyenet_yawpitch.npy
#       gt_yawpitch.npy

#     Returns per item:
#       screen_seq: (T,3,72,128)
#       eyenet_seq: (T,2)
#       gt_seq    : (T,2)
#     """
#     def __init__(self, root, split="train",
#                  seq_len=30,
#                  step_stride=30,
#                  max_steps=None):
#         self.root = root
#         self.split = split
#         self.seq_len = seq_len
#         self.step_stride = step_stride

#         if split.startswith("train"):
#             pat = os.path.join(root, "train*", "step*")
#         elif split.startswith("val"):
#             pat = os.path.join(root, "val*", "step*")
#         elif split.startswith("test"):
#             pat = os.path.join(root, "test*", "step*")
#         else:
#             raise ValueError(split)

#         self.step_dirs = sorted(glob.glob(pat))
#         if max_steps is not None:
#             self.step_dirs = self.step_dirs[:max_steps]

#         self._screen_cache = {}
#         self._eyenet_cache = {}
#         self._gt_cache = {}

#         # build (step_idx, start) index AFTER we know aligned lengths
#         self.index = []
#         for step_i, step_dir in enumerate(self.step_dirs):
#             e_path = os.path.join(step_dir, "eyenet_yawpitch.npy")
#             g_path = os.path.join(step_dir, "gt_yawpitch.npy")
#             if not (os.path.exists(e_path) and os.path.exists(g_path)):
#                 continue

#             # load once to know aligned length
#             screen, eyenet, gt = self._load_step(step_dir)
#             N = screen.shape[0]
#             if N < seq_len:
#                 continue
#             for start in range(0, N - seq_len + 1, step_stride):
#                 self.index.append((step_i, start))

#     def __len__(self):
#         return len(self.index)

#     def _load_step(self, step_dir):
#         if step_dir in self._screen_cache:
#             return (self._screen_cache[step_dir],
#                     self._eyenet_cache[step_dir],
#                     self._gt_cache[step_dir])

#         cap = cv2.VideoCapture(os.path.join(step_dir, "screen.128x72.mp4"))
#         frames = []
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = frame.astype(np.float32) / 255.0
#             frame = np.transpose(frame, (2, 0, 1))  # (3,72,128)
#             frames.append(frame)
#         cap.release()
#         screen = np.stack(frames, axis=0)  # (Ns,3,72,128)

#         eyenet = np.load(os.path.join(step_dir, "eyenet_yawpitch.npy"))  # (Ne,2)
#         gt = np.load(os.path.join(step_dir, "gt_yawpitch.npy"))          # (Ng,2)

#         # align lengths by truncating to min
#         N = min(screen.shape[0], eyenet.shape[0], gt.shape[0])
#         screen = screen[:N]
#         eyenet = eyenet[:N]
#         gt = gt[:N]

#         self._screen_cache[step_dir] = screen
#         self._eyenet_cache[step_dir] = eyenet
#         self._gt_cache[step_dir] = gt
#         return screen, eyenet, gt

#     def __getitem__(self, idx):
#         step_i, start = self.index[idx]
#         step_dir = self.step_dirs[step_i]

#         screen, eyenet, gt = self._load_step(step_dir)
#         end = start + self.seq_len

#         screen_seq = torch.from_numpy(screen[start:end])  # (T,3,72,128)
#         eyenet_seq = torch.from_numpy(eyenet[start:end])  # (T,2)
#         gt_seq = torch.from_numpy(gt[start:end])          # (T,2)

#         return {
#             "screen_seq": screen_seq,
#             "eyenet_seq": eyenet_seq,
#             "gt_yawpitch_seq": gt_seq,
#         }

# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import cv2
# import random


# def augment_screen_sequence(screen_seq_np):
#     """
#     screen_seq_np: (T,3,72,128) float32 in [0,1]
#     returns augmented sequence with same shape.
#     Only used for training.
#     """
#     T, C, H, W = screen_seq_np.shape
#     # convert to (T,H,W,C) uint8 for OpenCV ops
#     seq = (screen_seq_np.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)

#     # --- random horizontal flip ---
#     if random.random() < 0.5:
#         seq = np.flip(seq, axis=2)  # flip W

#     # --- random small rotation + scale ---
#     angle = random.uniform(-5.0, 5.0)
#     scale = random.uniform(0.95, 1.05)
#     center = (W / 2.0, H / 2.0)
#     M = cv2.getRotationMatrix2D(center, angle, scale)

#     for t in range(T):
#         seq[t] = cv2.warpAffine(
#             seq[t],
#             M,
#             (W, H),
#             flags=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_REFLECT_101,
#         )

#     # --- brightness / contrast jitter ---
#     alpha = random.uniform(0.9, 1.1)  # contrast
#     beta = random.uniform(-10, 10)    # brightness (in 0–255 space)
#     seq = cv2.convertScaleAbs(seq, alpha=alpha, beta=beta)

#     # back to (T,3,72,128) in [0,1]
#     seq = seq.astype(np.float32) / 255.0
#     seq = seq.transpose(0, 3, 1, 2)
#     return seq


# class GazeRefineYawPitchDataset(Dataset):
#     """
#     Uses:
#       screen.128x72.mp4
#       eyenet_yawpitch.npy
#       gt_yawpitch.npy

#     Returns per item:
#       screen_seq: (T,3,72,128)
#       eyenet_seq: (T,2)
#       gt_seq    : (T,2)
#     """
#     def __init__(self, root, split="train",
#                  seq_len=30,
#                  step_stride=30,
#                  max_steps=None,
#                  augment=False):
#         self.root = root
#         self.split = split
#         self.seq_len = seq_len
#         self.step_stride = step_stride
#         self.augment = augment and split.startswith("train")

#         if split.startswith("train"):
#             pat = os.path.join(root, "train*", "step*")
#         elif split.startswith("val"):
#             pat = os.path.join(root, "val*", "step*")
#         elif split.startswith("test"):
#             pat = os.path.join(root, "test*", "step*")
#         else:
#             raise ValueError(split)

#         self.step_dirs = sorted(glob.glob(pat))
#         if max_steps is not None:
#             self.step_dirs = self.step_dirs[:max_steps]

#         self._screen_cache = {}
#         self._eyenet_cache = {}
#         self._gt_cache = {}

#         # build (step_idx, start) index AFTER we know aligned lengths
#         self.index = []
#         for step_i, step_dir in enumerate(self.step_dirs):
#             e_path = os.path.join(step_dir, "eyenet_yawpitch.npy")
#             g_path = os.path.join(step_dir, "gt_yawpitch.npy")
#             if not (os.path.exists(e_path) and os.path.exists(g_path)):
#                 continue

#             screen, eyenet, gt = self._load_step(step_dir)
#             N = screen.shape[0]
#             if N < seq_len:
#                 continue
#             for start in range(0, N - seq_len + 1, step_stride):
#                 self.index.append((step_i, start))

#     def __len__(self):
#         return len(self.index)

#     def _load_step(self, step_dir):
#         if step_dir in self._screen_cache:
#             return (self._screen_cache[step_dir],
#                     self._eyenet_cache[step_dir],
#                     self._gt_cache[step_dir])

#         cap = cv2.VideoCapture(os.path.join(step_dir, "screen.128x72.mp4"))
#         frames = []
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = frame.astype(np.float32) / 255.0
#             frame = np.transpose(frame, (2, 0, 1))  # (3,72,128)
#             frames.append(frame)
#         cap.release()
#         screen = np.stack(frames, axis=0)  # (Ns,3,72,128)

#         eyenet = np.load(os.path.join(step_dir, "eyenet_yawpitch.npy"))
#         gt = np.load(os.path.join(step_dir, "gt_yawpitch.npy"))

#         # align lengths by truncating to min
#         N = min(screen.shape[0], eyenet.shape[0], gt.shape[0])
#         screen = screen[:N]
#         eyenet = eyenet[:N]
#         gt = gt[:N]

#         self._screen_cache[step_dir] = screen
#         self._eyenet_cache[step_dir] = eyenet
#         self._gt_cache[step_dir] = gt
#         return screen, eyenet, gt

#     def __getitem__(self, idx):
#         step_i, start = self.index[idx]
#         step_dir = self.step_dirs[step_i]

#         screen, eyenet, gt = self._load_step(step_dir)
#         end = start + self.seq_len

#         screen_seq = screen[start:end]   # (T,3,72,128)
#         if self.augment:
#             screen_seq = augment_screen_sequence(screen_seq)

#         screen_seq = torch.from_numpy(screen_seq)
#         eyenet_seq = torch.from_numpy(eyenet[start:end])  # (T,2)
#         gt_seq = torch.from_numpy(gt[start:end])          # (T,2)

#         return {
#             "screen_seq": screen_seq,
#             "eyenet_seq": eyenet_seq,
#             "gt_yawpitch_seq": gt_seq,
#         }

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random


def augment_screen_sequence(screen_seq_np):
    """
    screen_seq_np: (T,3,72,128) float32 in [0,1]
    returns augmented sequence with same shape.
    Only used for training.
    """
    T, C, H, W = screen_seq_np.shape
    # (T,H,W,C) uint8 for OpenCV
    seq = (screen_seq_np.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)

    # --- random horizontal flip ---
    if random.random() < 0.5:
        seq = np.flip(seq, axis=2)  # flip W

    # --- random small rotation + scale ---
    angle = random.uniform(-5.0, 5.0)
    scale = random.uniform(0.95, 1.05)
    center = (W / 2.0, H / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    for t in range(T):
        seq[t] = cv2.warpAffine(
            seq[t],
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # --- brightness / contrast jitter ---
    alpha = random.uniform(0.9, 1.1)  # contrast
    beta = random.uniform(-10, 10)    # brightness
    seq = cv2.convertScaleAbs(seq, alpha=alpha, beta=beta)

    # back to (T,3,72,128) in [0,1]
    seq = seq.astype(np.float32) / 255.0
    seq = seq.transpose(0, 3, 1, 2)
    return seq


class GazeRefineYawPitchDataset(Dataset):
    """
    Uses:
      screen.128x72.mp4
      eyenet_yawpitch.npy
      gt_yawpitch.npy

    Returns per item:
      screen_seq: (T,3,72,128)
      eyenet_seq: (T,2)
      gt_seq    : (T,2)
    """
    def __init__(self, root, split="train",
                 seq_len=30,
                 step_stride=30,
                 max_steps=None,
                 augment=False,
                 step_dirs=None):
        self.root = root
        self.split = split
        self.seq_len = seq_len
        self.step_stride = step_stride
        self.augment = augment and split.startswith("train")

        # if step_dirs given (for k‑fold), use that; otherwise infer from split
        if step_dirs is not None:
            self.step_dirs = sorted(step_dirs)
        else:
            if split.startswith("train"):
                pat = os.path.join(root, "train*", "step*")
            elif split.startswith("val"):
                pat = os.path.join(root, "val*", "step*")
            elif split.startswith("test"):
                pat = os.path.join(root, "test*", "step*")
            else:
                raise ValueError(split)
            self.step_dirs = sorted(glob.glob(pat))

        if max_steps is not None:
            self.step_dirs = self.step_dirs[:max_steps]

        self._screen_cache = {}
        self._eyenet_cache = {}
        self._gt_cache = {}

        # build (step_idx, start) index AFTER we know aligned lengths
        self.index = []
        for step_i, step_dir in enumerate(self.step_dirs):
            e_path = os.path.join(step_dir, "eyenet_yawpitch.npy")
            g_path = os.path.join(step_dir, "gt_yawpitch.npy")
            if not (os.path.exists(e_path) and os.path.exists(g_path)):
                continue

            screen, eyenet, gt = self._load_step(step_dir)
            N = screen.shape[0]
            if N < seq_len:
                continue
            for start in range(0, N - seq_len + 1, step_stride):
                self.index.append((step_i, start))

    def __len__(self):
        return len(self.index)

    def _load_step(self, step_dir):
        if step_dir in self._screen_cache:
            return (self._screen_cache[step_dir],
                    self._eyenet_cache[step_dir],
                    self._gt_cache[step_dir])

        cap = cv2.VideoCapture(os.path.join(step_dir, "screen.128x72.mp4"))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))  # (3,72,128)
            frames.append(frame)
        cap.release()
        screen = np.stack(frames, axis=0)  # (Ns,3,72,128)

        eyenet = np.load(os.path.join(step_dir, "eyenet_yawpitch.npy"))
        gt = np.load(os.path.join(step_dir, "gt_yawpitch.npy"))

        # align lengths by truncating to min
        N = min(screen.shape[0], eyenet.shape[0], gt.shape[0])
        screen = screen[:N]
        eyenet = eyenet[:N]
        gt = gt[:N]

        self._screen_cache[step_dir] = screen
        self._eyenet_cache[step_dir] = eyenet
        self._gt_cache[step_dir] = gt
        return screen, eyenet, gt

    # def __getitem__(self, idx):
    #     step_i, start = self.index[idx]
    #     step_dir = self.step_dirs[step_i]

    #     screen, eyenet, gt = self._load_step(step_dir)
    #     end = start + self.seq_len

    #     screen_seq = screen[start:end]   # (T,3,72,128)
    #     if self.augment:
    #         screen_seq = augment_screen_sequence(screen_seq)

    #     screen_seq = torch.from_numpy(screen_seq)
    #     eyenet_seq = torch.from_numpy(eyenet[start:end])  # (T,2)
    #     gt_seq = torch.from_numpy(gt[start:end])          # (T,2)

    #     return {
    #         "screen_seq": screen_seq,
    #         "eyenet_seq": eyenet_seq,
    #         "gt_yawpitch_seq": gt_seq,
    #     }
    def __getitem__(self, idx):
        step_i, start = self.index[idx]
        step_dir = self.step_dirs[step_i]

        screen, eyenet, gt = self._load_step(step_dir)
        end = start + self.seq_len

        screen_seq = torch.from_numpy(screen[start:end])   # (T,3,72,128)
        eyenet_seq = torch.from_numpy(eyenet[start:end])   # (T,2)
        gt_seq     = torch.from_numpy(gt[start:end])       # (T,2)

        return {
            "screen_seq": screen_seq,
            "eyenet_seq": eyenet_seq,
            "gt_yawpitch_seq": gt_seq,
            "step_dir": step_dir,
            "start_idx": start,
        }