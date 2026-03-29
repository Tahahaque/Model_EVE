import os
import glob
import cv2
import numpy as np

root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"  # adjust

def list_step_dirs(root, split="train"):
    if split == "train":
        pat = os.path.join(root, "train*", "step*")
    elif split == "val":
        pat = os.path.join(root, "val*", "step*")
    elif split == "test":
        pat = os.path.join(root, "test*", "step*")
    else:
        raise ValueError(split)
    return sorted(glob.glob(pat))

for split in ["train", "val"]:
    step_dirs = list_step_dirs(root, split)
    for step_dir in step_dirs:
        mp4_path = os.path.join(step_dir, "basler_eyes.mp4")
        out_dir = os.path.join(step_dir, "basler_eyes_npy")
        if not os.path.isfile(mp4_path):
            continue
        if os.path.isdir(out_dir):
            # already extracted
            continue

        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(mp4_path)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            np.save(os.path.join(out_dir, f"{idx:06d}.npy"), gray)
            idx += 1
        cap.release()
        print(f"[{split}] {step_dir}: saved {idx} frames to {out_dir}")
