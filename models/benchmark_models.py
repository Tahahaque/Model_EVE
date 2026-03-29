# benchmark_models.py
# import time
# import math
# import os

# import torch
# from torch.utils.data import DataLoader

# from model_gru_temp import EyeNetGRU, EVEEyeSequenceDataset, mean_angular_error_deg
# from model_gazerefine_yawpitch_gru import GazeRefineYawPitchGRU
# from dataset_gazerefine_yawpitch import GazeRefineYawPitchDataset

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def count_parameters(model, only_trainable=True):
#     if only_trainable:
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#     else:
#         return sum(p.numel() for p in model.parameters())

# @torch.no_grad()
# def measure_inference_time(model, loader, device=DEVICE, max_batches=20):
#     model.to(device)
#     model.eval()

#     times = []
#     n_frames = 0

#     # small warmup to stabilize GPU timing
#     warmup = 2
#     seen = 0
#     for batch in loader:
#         if seen >= warmup:
#             break
#         seen += 1
#         if isinstance(batch, (list, tuple)):
#             imgs = batch[0].to(device)
#         else:
#             imgs = batch["screen_seq"].to(device)
#         _ = model(imgs) if not isinstance(batch, dict) else model(
#             batch["screen_seq"].to(device),
#             batch["eyenet_seq"].to(device)
#         )

#     # real timing
#     n_batches = 0
#     for batch in loader:
#         if max_batches is not None and n_batches >= max_batches:
#             break

#         if isinstance(batch, (list, tuple)):
#             imgs = batch[0].to(device)             # (B,T,1,H,W)
#         else:
#             imgs = batch["screen_seq"].to(device)  # (B,T,3,72,128)
#         B = imgs.size(0)

#         if device.type == "cuda":
#             torch.cuda.synchronize()
#         start = time.time()

#         _ = model(imgs) if not isinstance(batch, dict) else model(
#             batch["screen_seq"].to(device),
#             batch["eyenet_seq"].to(device)
#         )

#         if device.type == "cuda":
#             torch.cuda.synchronize()
#         end = time.time()

#         elapsed = end - start
#         times.append(elapsed)
#         n_frames += B
#         n_batches += 1

#     if len(times) == 0:
#         return math.nan, math.nan

#     avg_batch_time = sum(times) / len(times)
#     avg_time_per_seq = avg_batch_time / (n_frames / n_batches)
#     fps = 1.0 / avg_time_per_seq if avg_time_per_seq > 0 else math.nan
#     return avg_time_per_seq, fps

# def main():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
#     os.makedirs("results", exist_ok=True)

#     # 1) EyeNet-GRU benchmark on val split
#     val_eye = EVEEyeSequenceDataset(
#         root=root,
#         split="val",
#         camera="basler",
#         which_eye="left",
#         img_size=(64, 64),
#         seq_len=30,
#         step_stride=30,
#         max_steps=50,       # limit for speed
#     )
#     val_eye_loader = DataLoader(val_eye, batch_size=4, shuffle=False, num_workers=0)

#     eyenet = EyeNetGRU(out_gaze_dim=2, hidden_dim=256)
#     ckpt_path = "checkpoints/eyenet_gru_epoch5.pth"
#     ckpt = torch.load(ckpt_path, map_location=DEVICE)
#     eyenet.load_state_dict(ckpt["model_state_dict"])
#     eyenet.to(DEVICE)

#     eyenet_params = count_parameters(eyenet)
#     eye_time, eye_fps = measure_inference_time(eyenet, val_eye_loader)

#     print(f"EyeNet-GRU: params={eyenet_params}, avg_seq_time={eye_time:.6f}s, fps={eye_fps:.2f}")

#     # 2) GazeRefine-GRU benchmark on val01 split
#     val_refine = GazeRefineYawPitchDataset(
#         root=root,
#         split="val01",
#         seq_len=30,
#         step_stride=30,
#     )
#     val_refine_loader = DataLoader(val_refine, batch_size=4, shuffle=False, num_workers=0)

#     refine = GazeRefineYawPitchGRU(
#         cnn_out_channels=64,
#         gru_hidden_size=128,
#         gru_layers=1,
#     )
#     refine_ckpt = torch.load(
#         "checkpoints/gazerefine_yawpitch_gru_best.pth",
#         map_location=DEVICE
#     )
#     refine.load_state_dict(refine_ckpt["model_state_dict"])
#     refine.to(DEVICE)

#     refine_params = count_parameters(refine)
#     refine_time, refine_fps = measure_inference_time(refine, val_refine_loader)

#     print(f"GazeRefine-GRU: params={refine_params}, avg_seq_time={refine_time:.6f}s, fps={refine_fps:.2f}")

#     # 3) Save results as CSV for plotting
#     import csv
#     with open("results/model_benchmark.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["model", "parameters", "avg_seq_time_s", "fps"])
#         writer.writerow(["EyeNetGRU", eyenet_params, eye_time, eye_fps])
#         writer.writerow(["GazeRefineYawPitchGRU", refine_params, refine_time, refine_fps])

# if __name__ == "__main__":
#     main()

# benchmark_models.py
import time
import math
import os
import csv

import torch
from torch.utils.data import DataLoader

from model_gru_temp import EyeNetGRU, EVEEyeSequenceDataset
from model_gazerefine_yawpitch_gru import GazeRefineYawPitchGRU
from dataset_gazerefine_yawpitch import GazeRefineYawPitchDataset

# ---------- device ----------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("Using device:", DEVICE)


# ---------- helpers ----------
def count_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_inference_time(model, loader, max_batches=20):
    """
    Measure average inference time per sequence (batch element).
    """
    model.to(DEVICE)
    model.eval()

    times = []
    n_seqs = 0

    # small warmup to stabilize GPU timing
    warmup = 2
    seen = 0
    for batch in loader:
        if seen >= warmup:
            break
        seen += 1
        if isinstance(batch, (list, tuple)):
            # EyeNetGRU: batch = (imgs, gazes, pupils)
            imgs = batch[0].to(DEVICE)  # (B,T,1,H,W)
            _ = model(imgs)
        else:
            # GazeRefine: dict
            screen = batch["screen_seq"].to(DEVICE)  # (B,T,3,72,128)
            eyenet = batch["eyenet_seq"].to(DEVICE)  # (B,T,2)
            _ = model(screen, eyenet)

    # real timing
    n_batches = 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            imgs = batch[0].to(DEVICE)
            B = imgs.size(0)
        else:
            screen = batch["screen_seq"].to(DEVICE)
            eyenet = batch["eyenet_seq"].to(DEVICE)
            B = screen.size(0)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        if isinstance(batch, (list, tuple)):
            _ = model(imgs)
        else:
            _ = model(screen, eyenet)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        elapsed = end - start
        times.append(elapsed)
        n_seqs += B
        n_batches += 1

    if len(times) == 0:
        return math.nan, math.nan

    avg_batch_time = sum(times) / len(times)          # seconds per batch
    avg_time_per_seq = avg_batch_time / (n_seqs / n_batches)  # seconds per sequence
    fps = 1.0 / avg_time_per_seq if avg_time_per_seq > 0 else math.nan
    return avg_time_per_seq, fps


# ---------- main ----------
def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
    os.makedirs("results", exist_ok=True)

    # 1) EyeNet-GRU benchmark on val split
    val_eye = EVEEyeSequenceDataset(
        root=root,
        split="val",
        camera="basler",
        which_eye="left",
        img_size=(64, 64),
        seq_len=30,
        step_stride=30,
        max_steps=50,  # limit for speed
    )
    val_eye_loader = DataLoader(val_eye, batch_size=4, shuffle=False, num_workers=0)

    eyenet = EyeNetGRU(out_gaze_dim=2, hidden_dim=256)
    ckpt_path = "checkpoints/eyenet_gru_epoch5.pth"  # adjust if needed
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    eyenet.load_state_dict(ckpt["model_state_dict"])
    eyenet.to(DEVICE)

    eyenet_params = count_parameters(eyenet)
    eye_time, eye_fps = measure_inference_time(eyenet, val_eye_loader)

    print(f"EyeNet-GRU: params={eyenet_params}, "
          f"avg_seq_time={eye_time:.6f}s, fps={eye_fps:.2f}")

    # 2) GazeRefine-GRU benchmark on val01 split
    val_refine = GazeRefineYawPitchDataset(
        root=root,
        split="val01",
        seq_len=30,
        step_stride=30,
    )
    val_refine_loader = DataLoader(val_refine, batch_size=4, shuffle=False, num_workers=0)

    refine = GazeRefineYawPitchGRU(
        cnn_out_channels=64,
        gru_hidden_size=128,
        gru_layers=1,
    )
    refine_ckpt = torch.load(
        "checkpoints/gazerefine_yawpitch_gru_best.pth",
        map_location=DEVICE
    )
    refine.load_state_dict(refine_ckpt["model_state_dict"])
    refine.to(DEVICE)

    refine_params = count_parameters(refine)
    refine_time, refine_fps = measure_inference_time(refine, val_refine_loader)

    print(f"GazeRefine-GRU: params={refine_params}, "
          f"avg_seq_time={refine_time:.6f}s, fps={refine_fps:.2f}")

    # 3) Save results as CSV for plotting
    csv_path = "results/model_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "parameters", "avg_seq_time_s", "fps"])
        writer.writerow(["EyeNetGRU", eyenet_params, eye_time, eye_fps])
        writer.writerow(["GazeRefineYawPitchGRU", refine_params, refine_time, refine_fps])

    print("Saved benchmark results to", csv_path)


if __name__ == "__main__":
    main()