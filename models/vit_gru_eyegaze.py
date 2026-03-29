# import os
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
#     Small CNN to extract local features from 64x64 eye image.
#     Output: (B, 64, 16, 16)
#     """
#     def __init__(self, in_channels=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),          # 64 -> 32
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),          # 32 -> 16
#         )

#     def forward(self, x):
#         return self.net(x)            # (B,64,16,16)


# class PatchEmbeddingFromFeature(nn.Module):
#     """
#     Patch embedding on top of CNN feature map.
#     fmap_size=16, patch_size=4 -> 4x4=16 patches.
#     """
#     def __init__(self, fmap_size=16, patch_size=4,
#                  in_channels=64, embed_dim=192):
#         super().__init__()
#         assert fmap_size % patch_size == 0
#         self.fmap_size = fmap_size
#         self.patch_size = patch_size
#         self.num_patches = (fmap_size // patch_size) ** 2
#         self.proj = nn.Conv2d(
#             in_channels,
#             embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )

#     def forward(self, x):
#         # x: (B,C,H',W')
#         x = self.proj(x)          # (B,D,H'/P,W'/P)
#         x = x.flatten(2)          # (B,D,N)
#         x = x.transpose(1, 2)     # (B,N,D)
#         return x


# class MLP(nn.Module):
#     def __init__(self, dim, mlp_dim, drop=0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, mlp_dim)
#         self.fc2 = nn.Linear(mlp_dim, dim)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


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
#         x_norm = self.norm2(x)
#         x = x + self.mlp(x_norm)
#         return x


# class FrameCNNEyeViT(nn.Module):
#     """
#     CNN + ViT encoder for a single eye frame.
#     Input:  (B,1,64,64)
#     Output: (B, D) feature (CLS token)
#     """
#     def __init__(
#         self,
#         embed_dim=192,
#         depth=4,
#         num_heads=6,
#         mlp_dim=384,
#         drop=0.1,
#     ):
#         super().__init__()
#         self.cnn = CNNBackbone(in_channels=1)
#         self.patch_embed = PatchEmbeddingFromFeature(
#             fmap_size=16,
#             patch_size=4,
#             in_channels=64,
#             embed_dim=embed_dim,
#         )
#         num_patches = self.patch_embed.num_patches  # 16

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, 1 + num_patches, embed_dim)
#         )
#         self.pos_drop = nn.Dropout(drop)

#         self.blocks = nn.ModuleList([
#             TransformerEncoderBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_dim=mlp_dim,
#                 drop=drop,
#             )
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)

#     def forward(self, x):
#         # x: (B,1,64,64)
#         B = x.shape[0]
#         fmap = self.cnn(x)               # (B,64,16,16)
#         tokens = self.patch_embed(fmap)  # (B,N,D)
#         cls = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls, tokens], dim=1)  # (B,1+N,D)
#         x = x + self.pos_embed[:, : x.size(1), :]
#         x = self.pos_drop(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#         cls_out = x[:, 0]                # (B,D)
#         return cls_out


# # --------- ViT-GRU sequence model ---------

# class ViTGRUEyeGaze(nn.Module):
#     """
#     Per frame: CNN+ViT encoder -> feature D.
#     Sequence: GRU over T frames -> yaw,pitch per frame.

#     Input:  (B,T,1,64,64)
#     Output: gaze (B,T,2), pupil dummy (B,T,1) for compatibility.
#     """
#     def __init__(self,
#                  frame_embed_dim=192,
#                  gru_hidden_dim=256):
#         super().__init__()
#         self.frame_encoder = FrameCNNEyeViT(
#             embed_dim=frame_embed_dim,
#             depth=4,
#             num_heads=6,
#             mlp_dim=384,
#             drop=0.1,
#         )

#         self.gru = nn.GRU(
#             input_size=frame_embed_dim,
#             hidden_size=gru_hidden_dim,
#             num_layers=1,
#             batch_first=True,
#         )

#         self.gaze_head = nn.Linear(gru_hidden_dim, 2)
#         self.pupil_head = nn.Linear(gru_hidden_dim, 1)

#     def forward(self, x):
#         """
#         x: (B,T,1,64,64)
#         """
#         B, T, C, H, W = x.shape

#         x = x.view(B * T, C, H, W)
#         feats = self.frame_encoder(x)         # (B*T,D)
#         feats = feats.view(B, T, -1)          # (B,T,D)

#         gru_out, _ = self.gru(feats)         # (B,T,H)
#         gaze = self.gaze_head(gru_out)       # (B,T,2)
#         pupil = self.pupil_head(gru_out)     # (B,T,1)
#         return gaze, pupil


# # --------- Training / evaluation ---------

# def train_one_epoch(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     total_gaze = 0.0
#     total_pupil = 0.0

#     for imgs, gazes, pupils in loader:
#         imgs = imgs.to(device)        # (B,T,1,H,W)
#         gazes = gazes.to(device)      # (B,T,2)
#         pupils = pupils.to(device)    # (B,T,1)

#         pred_gaze, pred_pupil = model(imgs)
#         gaze_loss = F.mse_loss(pred_gaze, gazes)
#         pupil_loss = F.l1_loss(pred_pupil, pupils)
#         loss = gaze_loss + pupil_loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_gaze += gaze_loss.item()
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
#         imgs = imgs.to(device)
#         gazes = gazes.to(device)
#         pupils = pupils.to(device)

#         pred_gaze, pred_pupil = model(imgs)
#         gaze_loss = F.mse_loss(pred_gaze, gazes)
#         pupil_loss = F.l1_loss(pred_pupil, pupils)
#         loss = gaze_loss + pupil_loss

#         total_loss += loss.item()
#         total_pupil += pupil_loss.item()

#         all_pred.append(pred_gaze.reshape(-1, 2).cpu())
#         all_gt.append(gazes.reshape(-1, 2).cpu())

#     all_pred = torch.cat(all_pred, dim=0)
#     all_gt = torch.cat(all_gt, dim=0)
#     mae_deg = mean_angular_error_deg(all_pred, all_gt)

#     n = len(loader)
#     return total_loss / n, total_pupil / n, mae_deg


# def main():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     train_dataset = EVEEyeSequenceDataset(
#         root=root,
#         split="train",
#         camera="basler",
#         which_eye="left",
#         img_size=(64, 64),
#         seq_len=30,
#         step_stride=30,
#         max_steps=None
#     )
#     val_dataset = EVEEyeSequenceDataset(
#         root=root,
#         split="val",
#         camera="basler",
#         which_eye="left",
#         img_size=(64, 64),
#         seq_len=30,
#         step_stride=30,
#         max_steps=None
#     )

#     train_loader = DataLoader(train_dataset, batch_size=4,
#                               shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=4,
#                             shuffle=False, num_workers=0)

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print("Device:", device)

#     model = ViTGRUEyeGaze(
#         frame_embed_dim=192,
#         gru_hidden_dim=256,
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

#     num_epochs = 10
#     os.makedirs("checkpoints", exist_ok=True)

#     for epoch in range(num_epochs):
#         tr_loss, tr_gaze, tr_pupil = train_one_epoch(model, train_loader, optimizer, device)
#         val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)

#         torch.save(
#             {"model_state_dict": model.state_dict()},
#             f"checkpoints/vit_gru_epoch{epoch+1}.pth",
#         )

#         print(
#             f"[ViT-GRU] Epoch {epoch+1}/{num_epochs} "
#             f"train_loss={tr_loss:.4f} gazeMSE={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
#             f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
#             f"val_mean_angular_error={val_mae:.2f} deg"
#         )


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
    Small CNN to extract local features from 64x64 eye image.
    Output: (B, 64, 16, 16)
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 64 -> 32
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32 -> 16
        )

    def forward(self, x):
        return self.net(x)            # (B,64,16,16)


class PatchEmbeddingFromFeature(nn.Module):
    """
    Patch embedding on top of CNN feature map.
    fmap_size=16, patch_size=4 -> 4x4=16 patches.
    """
    def __init__(self, fmap_size=16, patch_size=4,
                 in_channels=64, embed_dim=192):
        super().__init__()
        assert fmap_size % patch_size == 0
        self.fmap_size = fmap_size
        self.patch_size = patch_size
        self.num_patches = (fmap_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B,C,H',W')
        x = self.proj(x)          # (B,D,H'/P,W'/P)
        x = x.flatten(2)          # (B,D,N)
        x = x.transpose(1, 2)     # (B,N,D)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class FrameCNNEyeViT(nn.Module):
    """
    CNN + ViT encoder for a single eye frame.
    Input:  (B,1,64,64)
    Output: (B, D) feature (CLS token)
    """
    def __init__(
        self,
        embed_dim=192,
        depth=4,
        num_heads=6,
        mlp_dim=384,
        drop=0.1,
    ):
        super().__init__()
        self.cnn = CNNBackbone(in_channels=1)
        self.patch_embed = PatchEmbeddingFromFeature(
            fmap_size=16,
            patch_size=4,
            in_channels=64,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches  # 16

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                drop=drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B,1,64,64)
        B = x.shape[0]
        fmap = self.cnn(x)               # (B,64,16,16)
        tokens = self.patch_embed(fmap)  # (B,N,D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B,1+N,D)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]                # (B,D)
        return cls_out


# --------- ViT-GRU sequence model ---------

class ViTGRUEyeGaze(nn.Module):
    """
    Per frame: CNN+ViT encoder -> feature D.
    Sequence: GRU over T frames -> yaw,pitch per frame.

    Input:  (B,T,1,64,64)
    Output: gaze (B,T,2), pupil (B,T,1)
    """
    def __init__(self,
                 frame_embed_dim=192,
                 gru_hidden_dim=256):
        super().__init__()
        self.frame_encoder = FrameCNNEyeViT(
            embed_dim=frame_embed_dim,
            depth=4,
            num_heads=6,
            mlp_dim=384,
            drop=0.1,
        )

        self.gru = nn.GRU(
            input_size=frame_embed_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.gaze_head = nn.Linear(gru_hidden_dim, 2)
        self.pupil_head = nn.Linear(gru_hidden_dim, 1)

    def forward(self, x):
        """
        x: (B,T,1,64,64)
        """
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.frame_encoder(x)         # (B*T,D)
        feats = feats.view(B, T, -1)          # (B,T,D)

        gru_out, _ = self.gru(feats)         # (B,T,H)
        gaze = self.gaze_head(gru_out)       # (B,T,2)
        pupil = self.pupil_head(gru_out)     # (B,T,1)
        return gaze, pupil


# --------- augmentation on eye sequences ---------

def augment_eyes_sequence(imgs_np):
    """
    imgs_np: (B,T,1,H,W) in [0,1], numpy
    returns augmented version with same shape.
    """
    imgs = (imgs_np * 255.0).astype("uint8")
    B, T, _, H, W = imgs.shape
    for b in range(B):
        do_flip = (torch.rand(1).item() < 0.5)
        alpha = float(torch.empty(1).uniform_(0.9, 1.1))
        beta = float(torch.empty(1).uniform_(-10, 10))
        for t in range(T):
            img = imgs[b, t, 0]
            if do_flip:
                img = cv2.flip(img, 1)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            imgs[b, t, 0] = img
    imgs = imgs.astype("float32") / 255.0
    return imgs


# --------- Training / evaluation ---------

def train_one_epoch(model, loader, optimizer, device, use_augment=True):
    model.train()
    total_loss = total_gaze = total_pupil = 0.0

    for imgs, gazes, pupils in loader:
        # imgs: (B,T,1,H,W) in [0,1]
        if use_augment:
            imgs_np = imgs.numpy()
            imgs_np = augment_eyes_sequence(imgs_np)
            imgs = torch.from_numpy(imgs_np)

        imgs = imgs.to(device)
        gazes = gazes.to(device)
        pupils = pupils.to(device)

        pred_gaze, pred_pupil = model(imgs)
        gaze_loss = F.mse_loss(pred_gaze, gazes)
        pupil_loss = F.l1_loss(pred_pupil, pupils)
        loss = gaze_loss + pupil_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_gaze += gaze_loss.item()
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
        imgs = imgs.to(device)
        gazes = gazes.to(device)
        pupils = pupils.to(device)

        pred_gaze, pred_pupil = model(imgs)
        gaze_loss = F.mse_loss(pred_gaze, gazes)
        pupil_loss = F.l1_loss(pred_pupil, pupils)
        loss = gaze_loss + pupil_loss

        total_loss += loss.item()
        total_pupil += pupil_loss.item()

        all_pred.append(pred_gaze.reshape(-1, 2).cpu())
        all_gt.append(gazes.reshape(-1, 2).cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    mae_deg = mean_angular_error_deg(all_pred, all_gt)

    n = len(loader)
    return total_loss / n, total_pupil / n, mae_deg


def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

    train_dataset = EVEEyeSequenceDataset(
        root=root,
        split="train",
        camera="basler",
        which_eye="left",
        img_size=(64, 64),
        seq_len=30,
        step_stride=30,
        max_steps=None
    )
    val_dataset = EVEEyeSequenceDataset(
        root=root,
        split="val",
        camera="basler",
        which_eye="left",
        img_size=(64, 64),
        seq_len=30,
        step_stride=30,
        max_steps=None
    )

    train_loader = DataLoader(train_dataset, batch_size=4,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = ViTGRUEyeGaze(
        frame_embed_dim=192,
        gru_hidden_dim=256,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    num_epochs = 10
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        # simple LR decay after half the epochs
        if epoch == 10:
            for g in optimizer.param_groups:
                g["lr"] *= 0.5

        tr_loss, tr_gaze, tr_pupil = train_one_epoch(
            model, train_loader, optimizer, device, use_augment=True
        )
        val_loss, val_pupil, val_mae = evaluate(model, val_loader, device)

        torch.save(
            {"model_state_dict": model.state_dict(),
             "epoch": epoch + 1,
             "val_mae": val_mae},
            f"checkpoints/vit_gru_epoch{epoch+1}.pth",
        )

        print(
            f"[ViT-GRU] Epoch {epoch+1}/{num_epochs} "
            f"train_loss={tr_loss:.4f} gazeMSE={tr_gaze:.4f} pupilL1={tr_pupil:.4f} | "
            f"val_loss={val_loss:.4f} val_pupilL1={val_pupil:.4f} "
            f"val_mean_angular_error={val_mae:.2f} deg"
        )


if __name__ == "__main__":
    main()