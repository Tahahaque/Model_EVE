# import os
# import time
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset

# from model_gru_temp import (
#     EVEEyeSequenceDataset,
#     mean_angular_error_deg,
#     list_step_dirs,
# )


# # =====================================================================
# # Dataset: eye + screen frames, aligned by frame index
# # =====================================================================

# class EVEEyeScreenSequenceDataset(Dataset):
#     """
#     Returns per item:
#       eye_imgs   : (T, 1, 64, 64)    grayscale eye crops
#       screen_seq : (T, 3, 72, 128)   RGB screen frames
#       gazes      : (T, 2)            yaw/pitch in radians
#       pupils     : (T, 1)            pupil diameter
#     """

#     def __init__(self, root, split="train",
#                  camera="basler", which_eye="left",
#                  eye_img_size=(64, 64),
#                  seq_len=30, step_stride=30,
#                  max_steps=None):
#         super().__init__()
#         self.root        = root
#         self.eye_img_size = eye_img_size

#         # Reuse EVEEyeSequenceDataset to enumerate valid sequences
#         self.eye_ds = EVEEyeSequenceDataset(
#             root=root, split=split, camera=camera,
#             which_eye=which_eye, img_size=eye_img_size,
#             seq_len=seq_len, step_stride=step_stride,
#             max_steps=max_steps,
#         )
#         self.sequences = self.eye_ds.sequences   # (npy_dir, idxs, gaze_seq, pupil_seq)

#         # Map npy_dir -> step_dir for loading screen video
#         self.npy_to_step = {}
#         for step_dir in list_step_dirs(root, split, max_steps=None):
#             npy_dir = os.path.abspath(
#                 os.path.join(step_dir, f"{camera}_eyes_npy")
#             )
#             if os.path.isdir(npy_dir):
#                 self.npy_to_step[npy_dir] = step_dir

#         self._screen_cache = {}

#     def __len__(self):
#         return len(self.sequences)

#     def _load_screen_frames(self, step_dir):
#         """Load and cache all screen frames for a step as float32 CHW."""
#         if step_dir in self._screen_cache:
#             return self._screen_cache[step_dir]

#         mp4_path = os.path.join(step_dir, "screen.128x72.mp4")
#         cap = cv2.VideoCapture(mp4_path)
#         frames = []
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             # BGR (H,W,3) -> RGB (3,H,W) float32
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = frame.astype(np.float32) / 255.0
#             frame = np.transpose(frame, (2, 0, 1))   # HWC -> CHW  (3,72,128)
#             frames.append(frame)
#         cap.release()

#         if len(frames) == 0:
#             frames = [np.zeros((3, 72, 128), dtype=np.float32)]
#         screen = np.stack(frames, axis=0)   # (N, 3, 72, 128)
#         self._screen_cache[step_dir] = screen
#         return screen

#     def __getitem__(self, idx):
#         npy_dir, idxs, gaze_seq, pupil_seq = self.sequences[idx]

#         # --- Eye frames ---
#         eye_imgs = []
#         for n in idxs:
#             gray = np.load(os.path.join(npy_dir, f"{int(n):06d}.npy"))
#             img  = cv2.resize(gray, self.eye_img_size,
#                               interpolation=cv2.INTER_AREA)
#             eye_imgs.append(img.astype(np.float32)[None] / 255.0)  # (1,H,W)
#         eye_imgs = np.stack(eye_imgs, axis=0)   # (T,1,H,W)

#         # --- Screen frames ---
#         step_dir   = self.npy_to_step[os.path.abspath(npy_dir)]
#         screen_all = self._load_screen_frames(step_dir)   # (N,3,72,128)
#         clipped    = idxs.clip(0, screen_all.shape[0] - 1)
#         screen_seq = screen_all[clipped]                  # (T,3,72,128)

#         eye_t    = torch.from_numpy(eye_imgs)
#         screen_t = torch.from_numpy(screen_seq)
#         gaze_t   = torch.from_numpy(gaze_seq.astype(np.float32))
#         pupil_t  = torch.from_numpy(
#             pupil_seq[:, None].astype(np.float32)         # (T,1)
#         )
#         return eye_t, screen_t, gaze_t, pupil_t


# # =====================================================================
# # Model components
# # =====================================================================

# class CNNBackboneEye(nn.Module):
#     """
#     3-block CNN for 64×64 grayscale eye image.
#     Output: (B, 128, 8, 8)
#     """
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1, 1),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                     # 64 -> 32

#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                     # 32 -> 16

#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                     # 16 -> 8
#         )

#     def forward(self, x):
#         return self.net(x)   # (B,128,8,8)


# class PatchEmbedding(nn.Module):
#     """Conv-based patch embedding from a CNN feature map."""
#     def __init__(self, fmap_size=8, patch_size=2,
#                  in_channels=128, embed_dim=256):
#         super().__init__()
#         assert fmap_size % patch_size == 0
#         self.num_patches = (fmap_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_channels, embed_dim,
#                               kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)            # (B,D,P,P)
#         return x.flatten(2).transpose(1, 2)  # (B,N,D)


# class MLP(nn.Module):
#     def __init__(self, dim, mlp_dim, drop=0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
#             nn.Linear(mlp_dim, dim), nn.Dropout(drop),
#         )

#     def forward(self, x):
#         return self.net(x)


# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_dim, drop=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn  = nn.MultiheadAttention(dim, num_heads,
#                                            dropout=drop, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp   = MLP(dim, mlp_dim, drop)

#     def forward(self, x):
#         a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
#         x = x + a
#         return x + self.mlp(self.norm2(x))


# class EyeViTEncoder(nn.Module):
#     """
#     CNN + ViT for a single 64×64 eye frame.
#     Output: (B, embed_dim)  — CLS token
#     """
#     def __init__(self, embed_dim=256, depth=4, num_heads=8,
#                  mlp_dim=512, drop=0.15):
#         super().__init__()
#         self.cnn         = CNNBackboneEye()
#         self.patch_embed = PatchEmbedding(8, 2, 128, embed_dim)
#         N = self.patch_embed.num_patches          # 16

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, embed_dim))
#         self.pos_drop  = nn.Dropout(drop)
#         self.blocks    = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads, mlp_dim, drop)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token,  std=0.02)

#     def forward(self, x):
#         B = x.shape[0]
#         tokens = self.patch_embed(self.cnn(x))        # (B,N,D)
#         x = torch.cat([self.cls_token.expand(B,-1,-1), tokens], 1)
#         x = self.pos_drop(x + self.pos_embed[:, :x.size(1)])
#         for blk in self.blocks:
#             x = blk(x)
#         return self.norm(x)[:, 0]                     # (B,D)


# class ScreenCNNEncoder(nn.Module):
#     """
#     3-block CNN + global pool for 128×72 RGB screen frame.
#     Output: (B, out_dim)
#     """
#     def __init__(self, out_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 32, 3, 1, 1),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                     # 72x128 -> 36x64

#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),                     # 36x64 -> 18x32

#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128), nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),             # (B,128,1,1)
#         )
#         self.proj = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, out_dim),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.proj(self.net(x))   # (B, out_dim)


# class ViTGRUEyeScreen(nn.Module):
#     """
#     EyeViT + ScreenCNN fused per-frame -> 2-layer bidir GRU -> gaze + pupil.

#     Input:
#       eye_seq    : (B, T, 1,  64,  64)
#       screen_seq : (B, T, 3,  72, 128)
#     Output:
#       gaze  : (B, T, 2)
#       pupil : (B, T, 1)
#     """
#     def __init__(self, eye_embed_dim=256, screen_embed_dim=128,
#                  gru_hidden_dim=256, gru_layers=2, drop=0.15):
#         super().__init__()
#         self.eye_encoder    = EyeViTEncoder(
#             embed_dim=eye_embed_dim, depth=4, num_heads=8,
#             mlp_dim=512, drop=drop,
#         )
#         self.screen_encoder = ScreenCNNEncoder(out_dim=screen_embed_dim)

#         fusion_dim = eye_embed_dim + screen_embed_dim

#         # Cross-modal fusion before GRU
#         self.fusion = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop),
#         )

#         self.gru = nn.GRU(
#             input_size=fusion_dim,
#             hidden_size=gru_hidden_dim,
#             num_layers=gru_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=drop if gru_layers > 1 else 0.0,
#         )
#         gru_out = gru_hidden_dim * 2   # bidirectional

#         self.gaze_head = nn.Sequential(
#             nn.Linear(gru_out, 128), nn.ReLU(inplace=True),
#             nn.Dropout(drop), nn.Linear(128, 2),
#         )
#         self.pupil_head = nn.Sequential(
#             nn.Linear(gru_out, 64), nn.ReLU(inplace=True),
#             nn.Linear(64, 1),
#         )

#     def forward(self, eye_seq, screen_seq):
#         B, T, _, H,  W  = eye_seq.shape
#         _, _, C, Hs, Ws = screen_seq.shape

#         eye_feats    = self.eye_encoder(
#             eye_seq.reshape(B*T, 1, H, W)
#         )                                         # (B*T, De)
#         screen_feats = self.screen_encoder(
#             screen_seq.reshape(B*T, C, Hs, Ws)
#         )                                         # (B*T, Ds)

#         fused = torch.cat([eye_feats, screen_feats], dim=1)  # (B*T, De+Ds)
#         fused = self.fusion(fused)
#         fused = fused.reshape(B, T, -1)           # (B,  T,  D)

#         gru_out, _ = self.gru(fused)             # (B,  T, 2H)
#         return self.gaze_head(gru_out), self.pupil_head(gru_out)


# # =====================================================================
# # Throughput benchmark
# # =====================================================================

# def benchmark_model(model, device, batch_size=4, seq_len=30,
#                     warmup=5, runs=30):
#     model.eval()
#     eye    = torch.randn(batch_size, seq_len, 1,  64,  64).to(device)
#     screen = torch.randn(batch_size, seq_len, 3,  72, 128).to(device)

#     with torch.no_grad():
#         for _ in range(warmup):
#             model(eye, screen)

#     if device.type == "cuda": torch.cuda.synchronize()
#     if device.type == "mps":  torch.mps.synchronize()

#     t0 = time.perf_counter()
#     with torch.no_grad():
#         for _ in range(runs):
#             model(eye, screen)
#     if device.type == "cuda": torch.cuda.synchronize()
#     if device.type == "mps":  torch.mps.synchronize()
#     elapsed = time.perf_counter() - t0

#     seqs_per_sec = (runs * batch_size) / elapsed
#     total_p   = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print("\n" + "=" * 60)
#     print("  ViT-GRU-EyeScreen Model Summary")
#     print("=" * 60)
#     print(f"  Total parameters    : {total_p:,}  ({total_p/1e6:.2f}M)")
#     print(f"  Trainable parameters: {trainable:,}  ({trainable/1e6:.2f}M)")
#     print(f"  Throughput          : {seqs_per_sec:.0f} sequences/sec")
#     print(f"  (batch={batch_size}, seq_len={seq_len}, device={device})")
#     print("=" * 60 + "\n")
#     return seqs_per_sec, total_p


# # =====================================================================
# # Augmentation
# # =====================================================================

# def augment_sequence(eye_np, screen_np):
#     """
#     eye_np   : (B, T, 1,  H,  W)  float32 [0,1]
#     screen_np: (B, T, 3, Hs, Ws)  float32 [0,1]
#     Applies consistent random transforms per batch item.
#     """
#     B, T, _, H, W   = eye_np.shape
#     eye_u   = (eye_np   * 255).astype("uint8")
#     scr_u   = (screen_np * 255).astype("uint8")

#     for b in range(B):
#         do_flip = torch.rand(1).item() < 0.5
#         # eye jitter
#         e_alpha = float(torch.empty(1).uniform_(0.8, 1.2))
#         e_beta  = float(torch.empty(1).uniform_(-15, 15))
#         do_blur = torch.rand(1).item() < 0.3
#         # screen colour jitter
#         s_alpha = float(torch.empty(1).uniform_(0.85, 1.15))
#         s_beta  = float(torch.empty(1).uniform_(-10, 10))

#         for t in range(T):
#             # --- eye ---
#             img = eye_u[b, t, 0]
#             if do_flip:
#                 img = cv2.flip(img, 1)
#             img = cv2.convertScaleAbs(img, alpha=e_alpha, beta=e_beta)
#             if do_blur:
#                 img = cv2.GaussianBlur(img, (3, 3), 0)
#             noise = (torch.randn(H, W) * 3).numpy().astype("int16")
#             img = (img.astype("int16") + noise).clip(0, 255).astype("uint8")
#             eye_u[b, t, 0] = img

#             # --- screen (flip consistent with eye) ---
#             sc = scr_u[b, t].transpose(1, 2, 0)   # CHW -> HWC
#             if do_flip:
#                 sc = cv2.flip(sc, 1)
#             sc = cv2.convertScaleAbs(sc, alpha=s_alpha, beta=s_beta)
#             scr_u[b, t] = np.ascontiguousarray(sc.transpose(2, 0, 1))  # HWC -> CHW

#     return eye_u.astype("float32") / 255.0, scr_u.astype("float32") / 255.0


# # =====================================================================
# # Angular loss (differentiable)
# # =====================================================================

# def angular_loss(pred, gt):
#     """pred, gt: (N,2) yaw/pitch radians -> scalar loss."""
#     def to_vec(a):
#         yaw, pitch = a[:, 0], a[:, 1]
#         return torch.stack([
#             torch.cos(pitch) * torch.sin(yaw),
#             torch.sin(pitch),
#             torch.cos(pitch) * torch.cos(yaw),
#         ], dim=1)
#     dot = (to_vec(pred) * to_vec(gt)).sum(1).clamp(-1+1e-6, 1-1e-6)
#     return torch.acos(dot).mean()


# # =====================================================================
# # Train / evaluate
# # =====================================================================

# def train_one_epoch(model, loader, optimizer, device,
#                     use_augment=True, gaze_weight=50.0):
#     model.train()
#     total_loss = total_gaze = total_pupil = 0.0

#     for eye_seq, screen_seq, gazes, pupils in loader:
#         if use_augment:
#             eye_np, scr_np = augment_sequence(
#                 eye_seq.numpy(), screen_seq.numpy()
#             )
#             eye_seq    = torch.from_numpy(eye_np).contiguous()
#             screen_seq = torch.from_numpy(scr_np).contiguous()

#         eye_seq    = eye_seq.to(device)
#         screen_seq = screen_seq.to(device)
#         gazes      = gazes.to(device)
#         pupils     = pupils.to(device)

#         pred_gaze, pred_pupil = model(eye_seq, screen_seq)

#         gaze_mse  = F.mse_loss(pred_gaze, gazes)
#         gaze_ang  = angular_loss(pred_gaze.reshape(-1,2), gazes.reshape(-1,2))
#         gaze_loss = gaze_mse + gaze_ang
#         pupil_loss = F.l1_loss(pred_pupil, pupils)

#         loss = gaze_weight * gaze_loss + pupil_loss

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         total_loss  += loss.item()
#         total_gaze  += gaze_loss.item()
#         total_pupil += pupil_loss.item()

#     n = len(loader)
#     return total_loss / n, total_gaze / n, total_pupil / n


# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     total_loss = total_pupil = 0.0
#     all_pred, all_gt = [], []

#     for eye_seq, screen_seq, gazes, pupils in loader:
#         eye_seq    = eye_seq.to(device)
#         screen_seq = screen_seq.to(device)
#         gazes      = gazes.to(device)
#         pupils     = pupils.to(device)

#         pred_gaze, pred_pupil = model(eye_seq, screen_seq)
#         gaze_loss  = F.mse_loss(pred_gaze, gazes)
#         pupil_loss = F.l1_loss(pred_pupil, pupils)
#         loss = gaze_loss + pupil_loss

#         total_loss  += loss.item()
#         total_pupil += pupil_loss.item()
#         all_pred.append(pred_gaze.reshape(-1, 2).cpu())
#         all_gt.append(gazes.reshape(-1, 2).cpu())

#     all_pred = torch.cat(all_pred)
#     all_gt   = torch.cat(all_gt)
#     mae_deg  = mean_angular_error_deg(all_pred, all_gt)

#     n = len(loader)
#     return total_loss / n, total_pupil / n, mae_deg


# # =====================================================================
# # Main
# # =====================================================================

# def main():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     common = dict(camera="basler", which_eye="left",
#                   eye_img_size=(64,64), seq_len=30, max_steps=None)

#     train_dataset = EVEEyeScreenSequenceDataset(
#         root=root, split="train", step_stride=15, **common   # overlapping windows
#     )
#     val_dataset = EVEEyeScreenSequenceDataset(
#         root=root, split="val",   step_stride=30, **common
#     )

#     train_loader = DataLoader(train_dataset, batch_size=4,
#                               shuffle=True,  num_workers=0)
#     val_loader   = DataLoader(val_dataset,   batch_size=4,
#                               shuffle=False, num_workers=0)

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print("Device:", device)

#     model = ViTGRUEyeScreen(
#         eye_embed_dim=256,
#         screen_embed_dim=128,
#         gru_hidden_dim=256,
#         gru_layers=2,
#         drop=0.15,
#     ).to(device)

#     # Print params + measure throughput before training
#     benchmark_model(model, device, batch_size=4, seq_len=30)

#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=3e-4, weight_decay=1e-3
#     )

#     num_epochs    = 5
#     warmup_epochs = 3
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5
#     )

#     os.makedirs("checkpoints", exist_ok=True)
#     best_mae = float("inf")

#     for epoch in range(num_epochs):
#         # Linear LR warmup
#         if epoch < warmup_epochs:
#             lr_scale = (epoch + 1) / warmup_epochs
#             for g in optimizer.param_groups:
#                 g["lr"] = 3e-4 * lr_scale

#         tr_loss, tr_gaze, tr_pupil = train_one_epoch(
#             model, train_loader, optimizer, device,
#             use_augment=True, gaze_weight=50.0,
#         )
#         val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)

#         if epoch >= warmup_epochs:
#             scheduler.step()

#         current_lr = optimizer.param_groups[0]["lr"]
#         is_best    = val_mae < best_mae

#         # Save best checkpoint
#         if is_best:
#             best_mae = val_mae
#             torch.save(
#                 {"model_state_dict": model.state_dict(),
#                  "epoch": epoch + 1, "val_mae": val_mae},
#                 "checkpoints/vit_gru_eye_screen_best.pth",
#             )

#         # Periodic checkpoint every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             torch.save(
#                 {"model_state_dict": model.state_dict(),
#                  "epoch": epoch + 1, "val_mae": val_mae},
#                 f"checkpoints/vit_gru_eye_screen_epoch{epoch+1}.pth",
#             )

#         print(
#             f"[ViT-GRU-EyeScreen] Epoch {epoch+1:02d}/{num_epochs} "
#             f"lr={current_lr:.2e} "
#             f"train_loss={tr_loss:.4f} gazeLoss={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
#             f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
#             f"val_mae={val_mae:.2f}° "
#             f"{'★ best' if is_best else ''}"
#         )

#     print(f"\nBest val MAE: {best_mae:.2f}°")


# if __name__ == "__main__":
#     main()

import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_gru_temp import (
    EVEEyeSequenceDataset,
    mean_angular_error_deg,
    list_step_dirs,
)


# =====================================================================
# Dataset: eye + screen frames, aligned by frame index
# =====================================================================

class EVEEyeScreenSequenceDataset(Dataset):
    """
    Returns per item:
      eye_imgs   : (T, 1, 64, 64)    grayscale eye crops
      screen_seq : (T, 3, 72, 128)   RGB screen frames
      gazes      : (T, 2)            yaw/pitch in radians
      pupils     : (T, 1)            pupil diameter
    """

    def __init__(self, root, split="train",
                 camera="basler", which_eye="left",
                 eye_img_size=(64, 64),
                 seq_len=30, step_stride=30,
                 max_steps=None):
        super().__init__()
        self.root        = root
        self.eye_img_size = eye_img_size

        # Reuse EVEEyeSequenceDataset to enumerate valid sequences
        self.eye_ds = EVEEyeSequenceDataset(
            root=root, split=split, camera=camera,
            which_eye=which_eye, img_size=eye_img_size,
            seq_len=seq_len, step_stride=step_stride,
            max_steps=max_steps,
        )
        self.sequences = self.eye_ds.sequences   # (npy_dir, idxs, gaze_seq, pupil_seq)

        # Map npy_dir -> step_dir for loading screen video
        self.npy_to_step = {}
        for step_dir in list_step_dirs(root, split, max_steps=None):
            npy_dir = os.path.abspath(
                os.path.join(step_dir, f"{camera}_eyes_npy")
            )
            if os.path.isdir(npy_dir):
                self.npy_to_step[npy_dir] = step_dir

        self._screen_cache = {}

    def __len__(self):
        return len(self.sequences)

    def _load_screen_frames(self, step_dir):
        """Load and cache all screen frames for a step as float32 CHW."""
        if step_dir in self._screen_cache:
            return self._screen_cache[step_dir]

        mp4_path = os.path.join(step_dir, "screen.128x72.mp4")
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # BGR (H,W,3) -> RGB (3,H,W) float32
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))   # HWC -> CHW  (3,72,128)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((3, 72, 128), dtype=np.float32)]
        screen = np.stack(frames, axis=0)   # (N, 3, 72, 128)
        self._screen_cache[step_dir] = screen
        return screen

    def __getitem__(self, idx):
        npy_dir, idxs, gaze_seq, pupil_seq = self.sequences[idx]

        # --- Eye frames ---
        eye_imgs = []
        for n in idxs:
            gray = np.load(os.path.join(npy_dir, f"{int(n):06d}.npy"))
            img  = cv2.resize(gray, self.eye_img_size,
                              interpolation=cv2.INTER_AREA)
            eye_imgs.append(img.astype(np.float32)[None] / 255.0)  # (1,H,W)
        eye_imgs = np.stack(eye_imgs, axis=0)   # (T,1,H,W)

        # --- Screen frames ---
        step_dir   = self.npy_to_step[os.path.abspath(npy_dir)]
        screen_all = self._load_screen_frames(step_dir)   # (N,3,72,128)
        clipped    = idxs.clip(0, screen_all.shape[0] - 1)
        screen_seq = screen_all[clipped]                  # (T,3,72,128)

        eye_t    = torch.from_numpy(eye_imgs)
        screen_t = torch.from_numpy(screen_seq)
        gaze_t   = torch.from_numpy(gaze_seq.astype(np.float32))
        pupil_t  = torch.from_numpy(
            pupil_seq[:, None].astype(np.float32)         # (T,1)
        )
        return eye_t, screen_t, gaze_t, pupil_t


# =====================================================================
# Model components
# =====================================================================

class CNNBackboneEye(nn.Module):
    """
    3-block CNN for 64×64 grayscale eye image.
    Output: (B, 128, 8, 8)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 64 -> 32

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 32 -> 16

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 16 -> 8
        )

    def forward(self, x):
        return self.net(x)   # (B,128,8,8)


class PatchEmbedding(nn.Module):
    """Conv-based patch embedding from a CNN feature map."""
    def __init__(self, fmap_size=8, patch_size=2,
                 in_channels=128, embed_dim=256):
        super().__init__()
        assert fmap_size % patch_size == 0
        self.num_patches = (fmap_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)            # (B,D,P,P)
        return x.flatten(2).transpose(1, 2)  # (B,N,D)


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_dim, drop)

    def forward(self, x):
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + a
        return x + self.mlp(self.norm2(x))


class EyeViTEncoder(nn.Module):
    """
    CNN + ViT for a single 64×64 eye frame.
    Output: (B, embed_dim)  — CLS token
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=8,
                 mlp_dim=512, drop=0.15):
        super().__init__()
        self.cnn         = CNNBackboneEye()
        self.patch_embed = PatchEmbedding(8, 2, 128, embed_dim)
        N = self.patch_embed.num_patches          # 16

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, embed_dim))
        self.pos_drop  = nn.Dropout(drop)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(self.cnn(x))        # (B,N,D)
        x = torch.cat([self.cls_token.expand(B,-1,-1), tokens], 1)
        x = self.pos_drop(x + self.pos_embed[:, :x.size(1)])
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]                     # (B,D)


class ScreenCNNEncoder(nn.Module):
    """
    4-block CNN + global pool for 128×72 RGB screen frame.
    Deeper network extracts richer content cues (text, UI, gaze target).
    Output: (B, out_dim)
    """
    def __init__(self, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 72x128 -> 36x64

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 36x64 -> 18x32

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 18x32 -> 9x16

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),             # (B,256,1,1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(self.net(x))   # (B, out_dim)


class ViTGRUEyeScreen(nn.Module):
    """
    EyeViT + ScreenCNN fused per-frame -> 2-layer bidir GRU -> gaze + pupil.

    Input:
      eye_seq    : (B, T, 1,  64,  64)
      screen_seq : (B, T, 3,  72, 128)
    Output:
      gaze  : (B, T, 2)
      pupil : (B, T, 1)
    """
    def __init__(self, eye_embed_dim=256, screen_embed_dim=192,
                 gru_hidden_dim=384, gru_layers=2, drop=0.15):
        super().__init__()
        self.eye_encoder    = EyeViTEncoder(
            embed_dim=eye_embed_dim, depth=4, num_heads=8,
            mlp_dim=512, drop=drop,
        )
        self.screen_encoder = ScreenCNNEncoder(out_dim=screen_embed_dim)

        fusion_dim = eye_embed_dim + screen_embed_dim

        # Cross-modal fusion before GRU
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if gru_layers > 1 else 0.0,
        )
        gru_out = gru_hidden_dim * 2   # bidirectional

        self.gaze_head = nn.Sequential(
            nn.Linear(gru_out, 128), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(128, 2),
        )
        self.pupil_head = nn.Sequential(
            nn.Linear(gru_out, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, eye_seq, screen_seq):
        B, T, _, H,  W  = eye_seq.shape
        _, _, C, Hs, Ws = screen_seq.shape

        eye_feats    = self.eye_encoder(
            eye_seq.reshape(B*T, 1, H, W)
        )                                         # (B*T, De)
        screen_feats = self.screen_encoder(
            screen_seq.reshape(B*T, C, Hs, Ws)
        )                                         # (B*T, Ds)

        fused = torch.cat([eye_feats, screen_feats], dim=1)  # (B*T, De+Ds)
        fused = self.fusion(fused)
        fused = fused.reshape(B, T, -1)           # (B,  T,  D)

        gru_out, _ = self.gru(fused)             # (B,  T, 2H)
        return self.gaze_head(gru_out), self.pupil_head(gru_out)


# =====================================================================
# Throughput benchmark
# =====================================================================

def benchmark_model(model, device, batch_size=4, seq_len=30,
                    warmup=5, runs=30):
    model.eval()
    eye    = torch.randn(batch_size, seq_len, 1,  64,  64).to(device)
    screen = torch.randn(batch_size, seq_len, 3,  72, 128).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(eye, screen)

    if device.type == "cuda": torch.cuda.synchronize()
    if device.type == "mps":  torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(eye, screen)
    if device.type == "cuda": torch.cuda.synchronize()
    if device.type == "mps":  torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    seqs_per_sec = (runs * batch_size) / elapsed
    total_p   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("  ViT-GRU-EyeScreen Model Summary")
    print("=" * 60)
    print(f"  Total parameters    : {total_p:,}  ({total_p/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable:,}  ({trainable/1e6:.2f}M)")
    print(f"  Throughput          : {seqs_per_sec:.0f} sequences/sec")
    print(f"  (batch={batch_size}, seq_len={seq_len}, device={device})")
    print("=" * 60 + "\n")
    return seqs_per_sec, total_p


# =====================================================================
# Augmentation
# =====================================================================

def augment_sequence(eye_np, screen_np):
    """
    eye_np   : (B, T, 1,  H,  W)  float32 [0,1]
    screen_np: (B, T, 3, Hs, Ws)  float32 [0,1]
    Applies consistent random transforms per batch item.
    """
    B, T, _, H, W   = eye_np.shape
    eye_u   = (eye_np   * 255).astype("uint8")
    scr_u   = (screen_np * 255).astype("uint8")

    for b in range(B):
        do_flip = torch.rand(1).item() < 0.5
        # eye jitter
        e_alpha = float(torch.empty(1).uniform_(0.8, 1.2))
        e_beta  = float(torch.empty(1).uniform_(-15, 15))
        do_blur = torch.rand(1).item() < 0.3
        # screen colour jitter
        s_alpha = float(torch.empty(1).uniform_(0.85, 1.15))
        s_beta  = float(torch.empty(1).uniform_(-10, 10))

        for t in range(T):
            # --- eye ---
            img = eye_u[b, t, 0]
            if do_flip:
                img = cv2.flip(img, 1)
            img = cv2.convertScaleAbs(img, alpha=e_alpha, beta=e_beta)
            if do_blur:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            noise = (torch.randn(H, W) * 3).numpy().astype("int16")
            img = (img.astype("int16") + noise).clip(0, 255).astype("uint8")
            eye_u[b, t, 0] = img

            # --- screen (flip consistent with eye) ---
            sc = scr_u[b, t].transpose(1, 2, 0)   # CHW -> HWC
            if do_flip:
                sc = cv2.flip(sc, 1)
            sc = cv2.convertScaleAbs(sc, alpha=s_alpha, beta=s_beta)
            scr_u[b, t] = np.ascontiguousarray(sc.transpose(2, 0, 1))  # HWC -> CHW

    return eye_u.astype("float32") / 255.0, scr_u.astype("float32") / 255.0


# =====================================================================
# Angular loss (differentiable)
# =====================================================================

def angular_loss(pred, gt):
    """pred, gt: (N,2) yaw/pitch radians -> scalar loss."""
    def to_vec(a):
        yaw, pitch = a[:, 0], a[:, 1]
        return torch.stack([
            torch.cos(pitch) * torch.sin(yaw),
            torch.sin(pitch),
            torch.cos(pitch) * torch.cos(yaw),
        ], dim=1)
    dot = (to_vec(pred) * to_vec(gt)).sum(1).clamp(-1+1e-6, 1-1e-6)
    return torch.acos(dot).mean()


# =====================================================================
# Train / evaluate
# =====================================================================

def train_one_epoch(model, loader, optimizer, device,
                    use_augment=True, gaze_weight=50.0):
    model.train()
    total_loss = total_gaze = total_pupil = 0.0

    for eye_seq, screen_seq, gazes, pupils in loader:
        if use_augment:
            eye_np, scr_np = augment_sequence(
                eye_seq.numpy(), screen_seq.numpy()
            )
            eye_seq    = torch.from_numpy(eye_np).contiguous()
            screen_seq = torch.from_numpy(scr_np).contiguous()

        eye_seq    = eye_seq.to(device)
        screen_seq = screen_seq.to(device)
        gazes      = gazes.to(device)
        pupils     = pupils.to(device)

        pred_gaze, pred_pupil = model(eye_seq, screen_seq)

        gaze_mse  = F.mse_loss(pred_gaze, gazes)
        gaze_ang  = angular_loss(pred_gaze.reshape(-1,2), gazes.reshape(-1,2))
        gaze_loss = gaze_mse + gaze_ang
        pupil_loss = F.l1_loss(pred_pupil, pupils)

        loss = gaze_weight * gaze_loss + pupil_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_gaze  += gaze_loss.item()
        total_pupil += pupil_loss.item()

    n = len(loader)
    return total_loss / n, total_gaze / n, total_pupil / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_pupil = 0.0
    all_pred, all_gt = [], []

    for eye_seq, screen_seq, gazes, pupils in loader:
        eye_seq    = eye_seq.to(device)
        screen_seq = screen_seq.to(device)
        gazes      = gazes.to(device)
        pupils     = pupils.to(device)

        pred_gaze, pred_pupil = model(eye_seq, screen_seq)
        gaze_loss  = F.mse_loss(pred_gaze, gazes)
        pupil_loss = F.l1_loss(pred_pupil, pupils)
        loss = gaze_loss + pupil_loss

        total_loss  += loss.item()
        total_pupil += pupil_loss.item()
        all_pred.append(pred_gaze.reshape(-1, 2).cpu())
        all_gt.append(gazes.reshape(-1, 2).cpu())

    all_pred = torch.cat(all_pred)
    all_gt   = torch.cat(all_gt)
    mae_deg  = mean_angular_error_deg(all_pred, all_gt)

    n = len(loader)
    return total_loss / n, total_pupil / n, mae_deg


# =====================================================================
# Main
# =====================================================================

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
    resume_ckpt = None  # old checkpoint is incompatible with upgraded architecture — training from scratch

    common = dict(camera="basler", which_eye="left",
                  eye_img_size=(64,64), seq_len=30, max_steps=None)

    train_dataset = EVEEyeScreenSequenceDataset(
        root=root, split="train", step_stride=15, **common
    )
    val_dataset = EVEEyeScreenSequenceDataset(
        root=root, split="val",   step_stride=30, **common
    )

    train_loader = DataLoader(train_dataset, batch_size=4,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=4,
                              shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = ViTGRUEyeScreen(
        eye_embed_dim=256,
        screen_embed_dim=192,   # upgraded
        gru_hidden_dim=384,     # upgraded
        gru_layers=2,
        drop=0.15,
    ).to(device)

    # Resume from best checkpoint if available
    start_epoch = 0
    best_mae    = float("inf")
    if resume_ckpt and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_mae    = ckpt.get("val_mae", float("inf"))
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {resume_ckpt}  (epoch {start_epoch}, val_mae={best_mae:.2f}°)")

    benchmark_model(model, device, batch_size=4, seq_len=30)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )

    # Cosine annealing with warm restarts — escapes local minima better
    # T_0=15: first restart after 15 epochs, T_mult=1: same period each cycle
    num_epochs    = 5
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=1e-5
    )

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        # Linear LR warmup only on fresh training
        if start_epoch == 0 and epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = 3e-4 * lr_scale
        else:
            scheduler.step(epoch - max(start_epoch, warmup_epochs))

        tr_loss, tr_gaze, tr_pupil = train_one_epoch(
            model, train_loader, optimizer, device,
            use_augment=True, gaze_weight=50.0,
        )
        val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        is_best    = val_mae < best_mae

        if is_best:
            best_mae = val_mae
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch + 1, "val_mae": val_mae},
                "checkpoints/vit_gru_eye_screen_best.pth",
            )

        if (epoch + 1) % 5 == 0:
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch + 1, "val_mae": val_mae},
                f"checkpoints/vit_gru_eye_screen_epoch{epoch+1}.pth",
            )

        print(
            f"[ViT-GRU-EyeScreen] Epoch {epoch+1:02d}/{num_epochs} "
            f"lr={current_lr:.2e} "
            f"train_loss={tr_loss:.4f} gazeLoss={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
            f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
            f"val_mae={val_mae:.2f}° "
            f"{'★ best' if is_best else ''}"
        )

    print(f"\nBest val MAE: {best_mae:.2f}°")


if __name__ == "__main__":
    main()