# prep_gazerefine_from_gru.py
import os
import glob
import numpy as np
import torch
import h5py
import cv2
# from model_eyenet_basler_mobile import EyeNetGRU  # or wherever your class lives
# from model_gru_temp import EVEEyeSequenceDataset   # import from the file you pasted

from model_gru_temp import EyeNetGRU, EVEEyeSequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path):
    model = EyeNetGRU(out_gaze_dim=2, hidden_dim=256)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

# def load_step_eye_npy(step_dir, img_size=(64, 64)):
#     npy_dir = os.path.join(step_dir, "basler_eyes_npy")
#     files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
#     frames = []
#     for path in files:
#         gray = np.load(path)                # (H0,W0)
#         frame_resized = cv2.resize(gray, img_size, interpolation=cv2.INTER_AREA)
#         img = frame_resized.astype(np.float32) / 255.0
#         frames.append(img[None, ...])       # (1,H,W)
#     arr = np.stack(frames, axis=0)          # (N,1,H,W)
#     return arr
def load_step_eye_npy(step_dir, img_size=(64, 64)):
    npy_dir = os.path.join(step_dir, "basler_eyes_npy")
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    if len(files) == 0:
        return None  # nothing here

    frames = []
    for path in files:
        gray = np.load(path)
        frame_resized = cv2.resize(gray, img_size, interpolation=cv2.INTER_AREA)
        img = frame_resized.astype(np.float32) / 255.0
        frames.append(img[None, ...])
    arr = np.stack(frames, axis=0)
    return arr

def load_step_gt_yaw_pitch(step_dir, which_eye="left"):
    h5_path = os.path.join(step_dir, "basler.h5")
    with h5py.File(h5_path, "r") as f:
        g_key = f"{which_eye}_g_tobii"
        gaze = np.array(f[g_key]["data"], dtype=np.float32)  # (N,2)
    return gaze

# @torch.no_grad()
# def run_step(model, step_dir, img_size=(64, 64), which_eye="left"):
#     eyes = load_step_eye_npy(step_dir, img_size=img_size)          # (N,1,H,W)
#     gt = load_step_gt_yaw_pitch(step_dir, which_eye=which_eye)     # (N,2)

#     N = eyes.shape[0]
#     batch = torch.from_numpy(eyes).float().unsqueeze(0).to(DEVICE)  # (1,N,1,H,W)
#     pred_gaze, _ = model(batch)                 # (1,N,2)
#     pred_gaze = pred_gaze.squeeze(0).cpu().numpy()  # (N,2)

#     np.save(os.path.join(step_dir, "eyenet_yawpitch.npy"),
#             pred_gaze.astype(np.float32))
#     np.save(os.path.join(step_dir, "gt_yawpitch.npy"),
#             gt.astype(np.float32))
@torch.no_grad()
def run_step(model, step_dir, img_size=(64, 64), which_eye="left"):
    eyes = load_step_eye_npy(step_dir, img_size=img_size)
    if eyes is None:
        print("  skipping (no basler_eyes_npy):", step_dir)
        return

    gt = load_step_gt_yaw_pitch(step_dir, which_eye=which_eye)
    N = eyes.shape[0]
    gt = gt[:N]  # in case gt is longer

    batch = torch.from_numpy(eyes).float().unsqueeze(0).to(DEVICE)
    pred_gaze, _ = model(batch)
    pred_gaze = pred_gaze.squeeze(0).cpu().numpy()

    np.save(os.path.join(step_dir, "eyenet_yawpitch.npy"),
            pred_gaze.astype(np.float32))
    np.save(os.path.join(step_dir, "gt_yawpitch.npy"),
            gt.astype(np.float32))

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"  # adjust
    splits = ["train01","train02","val01", "test01"]
    ckpt_path = "checkpoints/eyenet_gru_epoch5.pth"                # adjust

    model = load_model(ckpt_path)

    for split in splits:
        split_dir = os.path.join(root, split)
        step_dirs = sorted(
            d for d in glob.glob(os.path.join(split_dir, "step*"))
            if os.path.isdir(d)
        )
        for step_dir in step_dirs:
            print("Processing", step_dir)
            run_step(model, step_dir)

if __name__ == "__main__":
    main()
