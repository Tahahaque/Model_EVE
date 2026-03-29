# import os
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# from model_gru_temp import (
#     EVEEyeSequenceDataset,
#     mean_angular_error_deg,
# )


# # --------- CNN + ViT encoder for one frame ---------

# class CNNBackbone(nn.Module):
#     """
#     Deeper CNN to extract richer local features from 64x64 eye image.
#     Output: (B, 128, 8, 8)
#     """
#     def __init__(self, in_channels=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),          # 64 -> 32

#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),          # 32 -> 16

#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),          # 16 -> 8
#         )

#     def forward(self, x):
#         return self.net(x)            # (B, 128, 8, 8)


# class PatchEmbeddingFromFeature(nn.Module):
#     """
#     Patch embedding on top of CNN feature map.
#     fmap_size=8, patch_size=2 -> 4x4=16 patches.
#     """
#     def __init__(self, fmap_size=8, patch_size=2,
#                  in_channels=128, embed_dim=256):
#         super().__init__()
#         assert fmap_size % patch_size == 0
#         self.num_patches = (fmap_size // patch_size) ** 2
#         self.proj = nn.Conv2d(
#             in_channels,
#             embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )

#     def forward(self, x):
#         x = self.proj(x)          # (B, D, H'/P, W'/P)
#         x = x.flatten(2)          # (B, D, N)
#         x = x.transpose(1, 2)     # (B, N, D)
#         return x


# class MLP(nn.Module):
#     def __init__(self, dim, mlp_dim, drop=0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, mlp_dim)
#         self.fc2 = nn.Linear(mlp_dim, dim)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_dim, drop=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             dropout=drop,
#             batch_first=True,
#         )
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = MLP(dim, mlp_dim, drop=drop)

#     def forward(self, x):
#         x_norm = self.norm1(x)
#         attn_out, _ = self.attn(x_norm, x_norm, x_norm)
#         x = x + attn_out
#         x = x + self.mlp(self.norm2(x))
#         return x


# class FrameCNNEyeViT(nn.Module):
#     """
#     CNN + ViT encoder for a single eye frame.
#     Input:  (B, 1, 64, 64)
#     Output: (B, D) feature (CLS token)
#     """
#     def __init__(self, embed_dim=256, depth=4, num_heads=8,
#                  mlp_dim=512, drop=0.15):
#         super().__init__()
#         self.cnn = CNNBackbone(in_channels=1)
#         self.patch_embed = PatchEmbeddingFromFeature(
#             fmap_size=8, patch_size=2,
#             in_channels=128, embed_dim=embed_dim,
#         )
#         num_patches = self.patch_embed.num_patches  # 16

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, 1 + num_patches, embed_dim)
#         )
#         self.pos_drop = nn.Dropout(drop)

#         self.blocks = nn.ModuleList([
#             TransformerEncoderBlock(
#                 dim=embed_dim, num_heads=num_heads,
#                 mlp_dim=mlp_dim, drop=drop,
#             )
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)

#     def forward(self, x):
#         B = x.shape[0]
#         fmap = self.cnn(x)               # (B, 128, 8, 8)
#         tokens = self.patch_embed(fmap)  # (B, N, D)
#         cls = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls, tokens], dim=1)
#         x = x + self.pos_embed[:, :x.size(1), :]
#         x = self.pos_drop(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#         return x[:, 0]                   # CLS token: (B, D)


# # --------- ViT-GRU sequence model ---------

# class ViTGRUEyeGaze(nn.Module):
#     """
#     Per frame: CNN+ViT encoder -> feature D.
#     Sequence:  2-layer bidirectional GRU over T frames.

#     Input:  (B, T, 1, 64, 64)
#     Output: gaze (B, T, 2), pupil (B, T, 1)
#     """
#     def __init__(self, frame_embed_dim=256, gru_hidden_dim=256,
#                  gru_layers=2, drop=0.15):
#         super().__init__()
#         self.frame_encoder = FrameCNNEyeViT(
#             embed_dim=frame_embed_dim,
#             depth=4, num_heads=8,
#             mlp_dim=512, drop=drop,
#         )

#         # Bidirectional GRU: output dim = 2 * gru_hidden_dim
#         self.gru = nn.GRU(
#             input_size=frame_embed_dim,
#             hidden_size=gru_hidden_dim,
#             num_layers=gru_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=drop if gru_layers > 1 else 0.0,
#         )
#         gru_out_dim = gru_hidden_dim * 2  # bidirectional

#         self.gaze_head = nn.Sequential(
#             nn.Linear(gru_out_dim, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(drop),
#             nn.Linear(128, 2),
#         )
#         self.pupil_head = nn.Sequential(
#             nn.Linear(gru_out_dim, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, 1),
#         )

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x_flat = x.view(B * T, C, H, W)
#         feats = self.frame_encoder(x_flat)   # (B*T, D)
#         feats = feats.view(B, T, -1)         # (B, T, D)

#         gru_out, _ = self.gru(feats)         # (B, T, 2*H)
#         gaze = self.gaze_head(gru_out)       # (B, T, 2)
#         pupil = self.pupil_head(gru_out)     # (B, T, 1)
#         return gaze, pupil


# # --------- Augmentation ---------

# def augment_eyes_sequence(imgs_np):
#     """
#     imgs_np: (B, T, 1, H, W) in [0, 1], numpy
#     Applies per-sequence random flips, brightness/contrast jitter,
#     and Gaussian blur.
#     """
#     imgs = (imgs_np * 255.0).astype("uint8")
#     B, T, _, H, W = imgs.shape
#     for b in range(B):
#         do_flip   = (torch.rand(1).item() < 0.5)
#         alpha     = float(torch.empty(1).uniform_(0.8, 1.2))   # contrast
#         beta      = float(torch.empty(1).uniform_(-15, 15))    # brightness
#         do_blur   = (torch.rand(1).item() < 0.3)
#         for t in range(T):
#             img = imgs[b, t, 0]
#             if do_flip:
#                 img = cv2.flip(img, 1)
#             img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#             if do_blur:
#                 img = cv2.GaussianBlur(img, (3, 3), 0)
#             # random noise
#             noise = (torch.randn(H, W) * 3).numpy().astype("int16")
#             img = (img.astype("int16") + noise).clip(0, 255).astype("uint8")
#             imgs[b, t, 0] = img
#     return imgs.astype("float32") / 255.0


# # --------- Losses ---------

# def angular_loss(pred, gt):
#     """
#     pred, gt: (N, 2) yaw/pitch in radians.
#     Returns mean angular error in radians as a differentiable loss.
#     """
#     def to_vec(angles):
#         yaw, pitch = angles[:, 0], angles[:, 1]
#         x = torch.cos(pitch) * torch.sin(yaw)
#         y = torch.sin(pitch)
#         z = torch.cos(pitch) * torch.cos(yaw)
#         return torch.stack([x, y, z], dim=1)

#     v_pred = to_vec(pred)
#     v_gt   = to_vec(gt)
#     dot = (v_pred * v_gt).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
#     return torch.acos(dot).mean()


# # --------- Training / evaluation ---------

# def train_one_epoch(model, loader, optimizer, device,
#                     use_augment=True, gaze_weight=50.0):
#     """
#     gaze_weight: multiplier on gaze loss to balance against pupil L1.
#     50× compensates for the ~100× magnitude difference observed.
#     """
#     model.train()
#     total_loss = total_gaze = total_pupil = 0.0

#     for imgs, gazes, pupils in loader:
#         if use_augment:
#             imgs_np = imgs.numpy()
#             imgs_np = augment_eyes_sequence(imgs_np)
#             imgs = torch.from_numpy(imgs_np)

#         imgs   = imgs.to(device)
#         gazes  = gazes.to(device)
#         pupils = pupils.to(device)

#         pred_gaze, pred_pupil = model(imgs)

#         # MSE + angular loss combo for gaze
#         gaze_mse  = F.mse_loss(pred_gaze, gazes)
#         pred_flat = pred_gaze.reshape(-1, 2)
#         gt_flat   = gazes.reshape(-1, 2)
#         gaze_ang  = angular_loss(pred_flat, gt_flat)
#         gaze_loss = gaze_mse + gaze_ang

#         pupil_loss = F.l1_loss(pred_pupil, pupils)

#         # Weight gaze heavily so it isn't swamped by pupil loss
#         loss = gaze_weight * gaze_loss + pupil_loss

#         optimizer.zero_grad()
#         loss.backward()
#         # Gradient clipping prevents exploding gradients in GRU
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
#     total_loss = 0.0
#     total_pupil = 0.0
#     all_pred, all_gt = [], []

#     for imgs, gazes, pupils in loader:
#         imgs   = imgs.to(device)
#         gazes  = gazes.to(device)
#         pupils = pupils.to(device)

#         pred_gaze, pred_pupil = model(imgs)
#         gaze_loss  = F.mse_loss(pred_gaze, gazes)
#         pupil_loss = F.l1_loss(pred_pupil, pupils)
#         loss = gaze_loss + pupil_loss

#         total_loss  += loss.item()
#         total_pupil += pupil_loss.item()

#         all_pred.append(pred_gaze.reshape(-1, 2).cpu())
#         all_gt.append(gazes.reshape(-1, 2).cpu())

#     all_pred = torch.cat(all_pred, dim=0)
#     all_gt   = torch.cat(all_gt,   dim=0)
#     mae_deg  = mean_angular_error_deg(all_pred, all_gt)

#     n = len(loader)
#     return total_loss / n, total_pupil / n, mae_deg


# def main():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     train_dataset = EVEEyeSequenceDataset(
#         root=root, split="train", camera="basler",
#         which_eye="left", img_size=(64, 64),
#         seq_len=30, step_stride=15,   # ← overlapping windows = more data
#         max_steps=None,
#     )
#     val_dataset = EVEEyeSequenceDataset(
#         root=root, split="val", camera="basler",
#         which_eye="left", img_size=(64, 64),
#         seq_len=30, step_stride=30,
#         max_steps=None,
#     )

#     train_loader = DataLoader(train_dataset, batch_size=8,
#                               shuffle=True,  num_workers=0)
#     val_loader   = DataLoader(val_dataset,   batch_size=8,
#                               shuffle=False, num_workers=0)

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print("Device:", device)

#     model = ViTGRUEyeGaze(
#         frame_embed_dim=256,
#         gru_hidden_dim=256,
#         gru_layers=2,
#         drop=0.15,
#     ).to(device)

#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Trainable parameters: {num_params:,}")

#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=3e-4, weight_decay=1e-3
#     )

#     num_epochs = 5
#     # Cosine annealing with a short warmup
#     warmup_epochs = 3
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5
#     )

#     os.makedirs("checkpoints", exist_ok=True)
#     best_mae = float("inf")

#     for epoch in range(num_epochs):
#         # Linear warmup
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

#         # Save only the best checkpoint
#         if val_mae < best_mae:
#             best_mae = val_mae
#             torch.save(
#                 {"model_state_dict": model.state_dict(),
#                  "epoch": epoch + 1,
#                  "val_mae": val_mae},
#                 "checkpoints/vit_gru_best.pth",
#             )

#         # Also save every 5 epochs as a safety net
#         if (epoch + 1) % 5 == 0:
#             torch.save(
#                 {"model_state_dict": model.state_dict(),
#                  "epoch": epoch + 1,
#                  "val_mae": val_mae},
#                 f"checkpoints/vit_gru_epoch{epoch+1}.pth",
#             )

#         print(
#             f"[ViT-GRU] Epoch {epoch+1:02d}/{num_epochs} "
#             f"lr={current_lr:.2e} "
#             f"train_loss={tr_loss:.4f} gazeLoss={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
#             f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
#             f"val_mae={val_mae:.2f}° "
#             f"{'★ best' if val_mae == best_mae else ''}"
#         )

#     print(f"\nBest val MAE: {best_mae:.2f}°")


# if __name__ == "__main__":
#     main()
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_gru_temp import (
    EVEEyeSequenceDataset,
    mean_angular_error_deg,
)


# --------- CNN + ViT encoder for one frame ---------

class CNNBackbone(nn.Module):
    """
    Deeper CNN to extract richer local features from 64x64 eye image.
    Output: (B, 128, 8, 8)
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 64 -> 32

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32 -> 16

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16 -> 8
        )

    def forward(self, x):
        return self.net(x)            # (B, 128, 8, 8)


class PatchEmbeddingFromFeature(nn.Module):
    """
    Patch embedding on top of CNN feature map.
    fmap_size=8, patch_size=2 -> 4x4=16 patches.
    """
    def __init__(self, fmap_size=8, patch_size=2,
                 in_channels=128, embed_dim=256):
        super().__init__()
        assert fmap_size % patch_size == 0
        self.num_patches = (fmap_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)          # (B, D, H'/P, W'/P)
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, drop=drop)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FrameCNNEyeViT(nn.Module):
    """
    CNN + ViT encoder for a single eye frame.
    Input:  (B, 1, 64, 64)
    Output: (B, D) feature (CLS token)
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=8,
                 mlp_dim=512, drop=0.15):
        super().__init__()
        self.cnn = CNNBackbone(in_channels=1)
        self.patch_embed = PatchEmbeddingFromFeature(
            fmap_size=8, patch_size=2,
            in_channels=128, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches  # 16

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim, num_heads=num_heads,
                mlp_dim=mlp_dim, drop=drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        fmap = self.cnn(x)               # (B, 128, 8, 8)
        tokens = self.patch_embed(fmap)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                   # CLS token: (B, D)


# --------- ViT-GRU sequence model ---------

class ViTGRUEyeGaze(nn.Module):
    """
    Per frame: CNN+ViT encoder -> feature D.
    Sequence:  2-layer bidirectional GRU over T frames.

    Input:  (B, T, 1, 64, 64)
    Output: gaze (B, T, 2), pupil (B, T, 1)
    """
    def __init__(self, frame_embed_dim=256, gru_hidden_dim=256,
                 gru_layers=2, drop=0.15):
        super().__init__()
        self.frame_encoder = FrameCNNEyeViT(
            embed_dim=frame_embed_dim,
            depth=4, num_heads=8,
            mlp_dim=512, drop=drop,
        )

        # Bidirectional GRU: output dim = 2 * gru_hidden_dim
        self.gru = nn.GRU(
            input_size=frame_embed_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden_dim * 2  # bidirectional

        self.gaze_head = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, 2),
        )
        self.pupil_head = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.frame_encoder(x_flat)   # (B*T, D)
        feats = feats.view(B, T, -1)         # (B, T, D)

        gru_out, _ = self.gru(feats)         # (B, T, 2*H)
        gaze = self.gaze_head(gru_out)       # (B, T, 2)
        pupil = self.pupil_head(gru_out)     # (B, T, 1)
        return gaze, pupil


# --------- Augmentation ---------

def augment_eyes_sequence(imgs_np):
    """
    imgs_np: (B, T, 1, H, W) in [0, 1], numpy
    Applies per-sequence random flips, brightness/contrast jitter,
    and Gaussian blur.
    """
    imgs = (imgs_np * 255.0).astype("uint8")
    B, T, _, H, W = imgs.shape
    for b in range(B):
        do_flip   = (torch.rand(1).item() < 0.5)
        alpha     = float(torch.empty(1).uniform_(0.8, 1.2))   # contrast
        beta      = float(torch.empty(1).uniform_(-15, 15))    # brightness
        do_blur   = (torch.rand(1).item() < 0.3)
        for t in range(T):
            img = imgs[b, t, 0]
            if do_flip:
                img = cv2.flip(img, 1)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            if do_blur:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            # random noise
            noise = (torch.randn(H, W) * 3).numpy().astype("int16")
            img = (img.astype("int16") + noise).clip(0, 255).astype("uint8")
            imgs[b, t, 0] = img
    return imgs.astype("float32") / 255.0


# --------- Losses ---------

def angular_loss(pred, gt):
    """
    pred, gt: (N, 2) yaw/pitch in radians.
    Returns mean angular error in radians as a differentiable loss.
    """
    def to_vec(angles):
        yaw, pitch = angles[:, 0], angles[:, 1]
        x = torch.cos(pitch) * torch.sin(yaw)
        y = torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        return torch.stack([x, y, z], dim=1)

    v_pred = to_vec(pred)
    v_gt   = to_vec(gt)
    dot = (v_pred * v_gt).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(dot).mean()


# --------- Throughput Benchmark ---------

def benchmark_model(model, device, batch_size=8, seq_len=30,
                    warmup=5, runs=30):
    """
    Measures inference throughput in sequences/sec.
    Mirrors the methodology used for EyeNetGRU and GazeRefine-GRU.
    """
    model.eval()
    dummy = torch.randn(batch_size, seq_len, 1, 64, 64).to(device)

    # Warmup passes (not timed)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    # Synchronise before timing (important for MPS/CUDA)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    import time
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    seqs_per_sec = (runs * batch_size) / elapsed
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 55)
    print("  ViT-GRU Model Summary")
    print("=" * 55)
    print(f"  Total parameters    : {total_params:,}  ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable:,}  ({trainable/1e6:.2f}M)")
    print(f"  Throughput          : {seqs_per_sec:.0f} sequences/sec")
    print(f"  (batch={batch_size}, seq_len={seq_len}, device={device})")
    print("=" * 55 + "\n")
    return seqs_per_sec, total_params


# --------- Training / evaluation ---------

def train_one_epoch(model, loader, optimizer, device,
                    use_augment=True, gaze_weight=50.0):
    """
    gaze_weight: multiplier on gaze loss to balance against pupil L1.
    50× compensates for the ~100× magnitude difference observed.
    """
    model.train()
    total_loss = total_gaze = total_pupil = 0.0

    for imgs, gazes, pupils in loader:
        if use_augment:
            imgs_np = imgs.numpy()
            imgs_np = augment_eyes_sequence(imgs_np)
            imgs = torch.from_numpy(imgs_np)

        imgs   = imgs.to(device)
        gazes  = gazes.to(device)
        pupils = pupils.to(device)

        pred_gaze, pred_pupil = model(imgs)

        # MSE + angular loss combo for gaze
        gaze_mse  = F.mse_loss(pred_gaze, gazes)
        pred_flat = pred_gaze.reshape(-1, 2)
        gt_flat   = gazes.reshape(-1, 2)
        gaze_ang  = angular_loss(pred_flat, gt_flat)
        gaze_loss = gaze_mse + gaze_ang

        pupil_loss = F.l1_loss(pred_pupil, pupils)

        # Weight gaze heavily so it isn't swamped by pupil loss
        loss = gaze_weight * gaze_loss + pupil_loss

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in GRU
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
    total_loss = 0.0
    total_pupil = 0.0
    all_pred, all_gt = [], []

    for imgs, gazes, pupils in loader:
        imgs   = imgs.to(device)
        gazes  = gazes.to(device)
        pupils = pupils.to(device)

        pred_gaze, pred_pupil = model(imgs)
        gaze_loss  = F.mse_loss(pred_gaze, gazes)
        pupil_loss = F.l1_loss(pred_pupil, pupils)
        loss = gaze_loss + pupil_loss

        total_loss  += loss.item()
        total_pupil += pupil_loss.item()

        all_pred.append(pred_gaze.reshape(-1, 2).cpu())
        all_gt.append(gazes.reshape(-1, 2).cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)
    mae_deg  = mean_angular_error_deg(all_pred, all_gt)

    n = len(loader)
    return total_loss / n, total_pupil / n, mae_deg


def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

    train_dataset = EVEEyeSequenceDataset(
        root=root, split="train", camera="basler",
        which_eye="left", img_size=(64, 64),
        seq_len=30, step_stride=15,   # ← overlapping windows = more data
        max_steps=None,
    )
    val_dataset = EVEEyeSequenceDataset(
        root=root, split="val", camera="basler",
        which_eye="left", img_size=(64, 64),
        seq_len=30, step_stride=30,
        max_steps=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=8,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=8,
                              shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = ViTGRUEyeGaze(
        frame_embed_dim=256,
        gru_hidden_dim=256,
        gru_layers=2,
        drop=0.15,
    ).to(device)

    # Print parameter count + measure throughput before training
    benchmark_model(model, device, batch_size=8, seq_len=30)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )

    num_epochs = 30
    # Cosine annealing with a short warmup
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_mae = float("inf")

    for epoch in range(num_epochs):
        # Linear warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = 3e-4 * lr_scale

        tr_loss, tr_gaze, tr_pupil = train_one_epoch(
            model, train_loader, optimizer, device,
            use_augment=True, gaze_weight=50.0,
        )
        val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)

        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Save only the best checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "val_mae": val_mae},
                "checkpoints/vit_gru_best.pth",
            )

        # Also save every 5 epochs as a safety net
        if (epoch + 1) % 5 == 0:
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "val_mae": val_mae},
                f"checkpoints/vit_gru_epoch{epoch+1}.pth",
            )

        print(
            f"[ViT-GRU] Epoch {epoch+1:02d}/{num_epochs} "
            f"lr={current_lr:.2e} "
            f"train_loss={tr_loss:.4f} gazeLoss={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
            f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
            f"val_mae={val_mae:.2f}° "
            f"{'★ best' if val_mae == best_mae else ''}"
        )

    print(f"\nBest val MAE: {best_mae:.2f}°")


if __name__ == "__main__":
    main()