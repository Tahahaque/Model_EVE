import os
import glob
from sklearn.model_selection import KFold

def get_step_dirs_two_trains(root):
    step_dirs = []
    for split_prefix in ["train01", "train02"]:
        pattern = os.path.join(root, split_prefix, "step*")
        step_dirs.extend(
            d for d in glob.glob(pattern) if os.path.isdir(d)
        )
    step_dirs = sorted(step_dirs)
    return step_dirs

def make_kfold_splits_two_trains(root, k=5, seed=42):
    step_dirs = get_step_dirs_two_trains(root)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []
    for _, (train_idx, val_idx) in enumerate(kf.split(step_dirs)):
        train_steps = [step_dirs[i] for i in train_idx]
        val_steps = [step_dirs[i] for i in val_idx]
        folds.append({"train": train_steps, "val": val_steps})
    return folds