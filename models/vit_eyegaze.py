# import os
# import math
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# from model_gru_temp import EVEEyeSequenceDataset, mean_angular_error_deg


# # ------------- ViT components -------------

# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size=64, patch_size=8,
#                  in_channels=1, embed_dim=128):
#         super().__init__()
#         assert img_size % patch_size == 0
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2

#         self.proj = nn.Conv2d(
#             in_channels,
#             embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )

#     def forward(self, x):
#         # x: (B,C,H,W)
#         B, C, H, W = x.shape
#         assert H == self.img_size and W == self.img_size
#         x = self.proj(x)          # (B,D,H/P,W/P)
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


# class ViTEyeGaze(nn.Module):
#     """
#     Vision Transformer that takes SINGLE eye frames and regresses yaw,pitch.

#     Input to forward(): (B,1,64,64)
#     Output:            (B,2)  [yaw,pitch]
#     """
#     def __init__(
#         self,
#         img_size=64,
#         patch_size=8,
#         in_channels=1,
#         embed_dim=128,
#         depth=4,
#         num_heads=4,
#         mlp_dim=256,
#         drop=0.1,
#     ):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_channels=in_channels,
#             embed_dim=embed_dim,
#         )
#         num_patches = self.patch_embed.num_patches

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
#         self.head = nn.Linear(embed_dim, 2)  # yaw,pitch

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         nn.init.trunc_normal_(self.head.weight, std=0.02)
#         if self.head.bias is not None:
#             nn.init.zeros_(self.head.bias)

#     def forward(self, x):
#         """
#         x: (B,1,64,64)
#         """
#         B = x.shape[0]
#         x = self.patch_embed(x)                 # (B,N,D)
#         cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
#         x = torch.cat([cls, x], dim=1)          # (B,1+N,D)
#         x = x + self.pos_embed[:, : x.size(1), :]
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)

#         cls_out = x[:, 0]           # (B,D)
#         pred = self.head(cls_out)   # (B,2)
#         return pred


# # ------------- Training / eval + visualizations -------------

# def train_one_epoch(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0.0
#     for imgs, gazes, _ in loader:
#         # imgs: (B,T,1,H,W) -> treat each frame independently
#         B, T, C, H, W = imgs.shape
#         imgs = imgs.view(B * T, C, H, W).to(device)
#         gazes = gazes.view(B * T, 2).to(device)

#         optimizer.zero_grad()
#         pred = model(imgs)            # (B*T,2)
#         loss = F.mse_loss(pred, gazes)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     return total_loss / len(loader)


# @torch.no_grad()
# def evaluate_and_visualize(model, loader, device, out_dir="vit_viz"):
#     model.eval()
#     total_loss = 0.0
#     all_pred, all_gt = [], []

#     best_err = float("inf")
#     worst_err = -1.0
#     best_img = None
#     worst_img = None

#     os.makedirs(out_dir, exist_ok=True)

#     for imgs, gazes, _ in loader:
#         B, T, C, H, W = imgs.shape
#         imgs_flat = imgs.view(B * T, C, H, W).to(device)
#         gazes_flat = gazes.view(B * T, 2).to(device)

#         pred = model(imgs_flat)                 # (B*T,2)
#         loss = F.mse_loss(pred, gazes_flat)
#         total_loss += loss.item()

#         pred_cpu = pred.cpu()
#         gt_cpu = gazes_flat.cpu()

#         # angular error per frame
#         err = []
#         for i in range(pred_cpu.shape[0]):
#             e = mean_angular_error_deg(
#                 pred_cpu[i:i+1, :],
#                 gt_cpu[i:i+1, :]
#             )
#             err.append(e)
#         err = torch.tensor(err)

#         # track best / worst
#         imgs_np = imgs_flat.cpu().numpy()  # (B*T,1,H,W)
#         for i in range(err.shape[0]):
#             e = err[i].item()
#             img = imgs_np[i, 0]  # (H,W) gray in [0,1]
#             img_uint8 = (img * 255).clip(0, 255).astype("uint8")

#             if e < best_err:
#                 best_err = e
#                 best_img = img_uint8
#             if e > worst_err:
#                 worst_err = e
#                 worst_img = img_uint8

#         all_pred.append(pred_cpu)
#         all_gt.append(gt_cpu)

#     all_pred = torch.cat(all_pred, dim=0)
#     all_gt = torch.cat(all_gt, dim=0)
#     mae_deg = mean_angular_error_deg(all_pred, all_gt)

#     # save best and worst images
#     if best_img is not None:
#         cv2.imwrite(os.path.join(out_dir, f"best_eye_{best_err:.2f}deg.png"),
#                     best_img)
#     if worst_img is not None:
#         cv2.imwrite(os.path.join(out_dir, f"worst_eye_{worst_err:.2f}deg.png"),
#                     worst_img)

#     avg_loss = total_loss / len(loader)
#     return avg_loss, mae_deg, best_err, worst_err


# def main():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     # reuse your sequence dataset but ignore temporal order (flatten T)
#     train_ds = EVEEyeSequenceDataset(
#         root=root,
#         split="train",
#         camera="basler",
#         which_eye="left",
#         img_size=(64, 64),
#         seq_len=30,
#         step_stride=30,
#         max_steps=None
#     )
#     val_ds = EVEEyeSequenceDataset(
#         root=root,
#         split="val",
#         camera="basler",
#         which_eye="left",
#         img_size=(64, 64),
#         seq_len=30,
#         step_stride=30,
#         max_steps=None
#     )

#     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print("Device:", device)

#     model = ViTEyeGaze(
#         img_size=64,
#         patch_size=8,
#         in_channels=1,
#         embed_dim=128,
#         depth=4,
#         num_heads=4,
#         mlp_dim=256,
#         drop=0.1,
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

#     num_epochs = 5
#     for epoch in range(1, num_epochs + 1):
#         tr_loss = train_one_epoch(model, train_loader, optimizer, device)
#         val_loss, val_mae, best_err, worst_err = evaluate_and_visualize(
#             model, val_loader, device, out_dir="vit_viz"
#         )
#         print(
#             f"[ViT-EyeGaze] Epoch {epoch}/{num_epochs} "
#             f"train_MSE={tr_loss:.4f} | "
#             f"val_MSE={val_loss:.4f} val_MAE={val_mae:.2f} deg "
#             f"(best={best_err:.2f} deg, worst={worst_err:.2f} deg)"
#         )

#     # optional: save model
#     os.makedirs("checkpoints", exist_ok=True)
#     torch.save({"model_state_dict": model.state_dict()},
#                "checkpoints/vit_eyegaze.pth")


# if __name__ == "__main__":
#     main()

# import os
# import math
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# from model_gru_temp import EVEEyeSequenceDataset, mean_angular_error_deg


# # ------------- ViT components -------------

# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size=64, patch_size=8,
#                  in_channels=1, embed_dim=192):
#         super().__init__()
#         assert img_size % patch_size == 0
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2

#         self.proj = nn.Conv2d(
#             in_channels,
#             embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size and W == self.img_size
#         x = self.proj(x)          # (B,D,H/P,W/P)
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


# class ViTEyeGaze(nn.Module):
#     """
#     Vision Transformer that takes SINGLE eye frames and regresses yaw,pitch.

#     Input:  (B,1,64,64)
#     Output: (B,2)  [yaw,pitch]
#     """
#     def __init__(
#         self,
#         img_size=64,
#         patch_size=8,
#         in_channels=1,
#         embed_dim=192,
#         depth=6,
#         num_heads=6,
#         mlp_dim=384,
#         drop=0.1,
#     ):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_channels=in_channels,
#             embed_dim=embed_dim,
#         )
#         num_patches = self.patch_embed.num_patches

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
#         self.head = nn.Linear(embed_dim, 2)  # yaw,pitch

#         self._init_weights()

#     def _init_weights(self):
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         nn.init.trunc_normal_(self.head.weight, std=0.02)
#         if self.head.bias is not None:
#             nn.init.zeros_(self.head.bias)

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)                 # (B,N,D)
#         cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
#         x = torch.cat([cls, x], dim=1)          # (B,1+N,D)
#         x = x + self.pos_embed[:, : x.size(1), :]
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)

#         cls_out = x[:, 0]           # (B,D)
#         pred = self.head(cls_out)   # (B,2)
#         return pred


# # ------------- simple eye augmentation -------------

# def augment_eyes(imgs_np):
#     """
#     imgs_np: (B*T,1,H,W) in [0,1], numpy
#     returns augmented version with same shape.
#     """
#     imgs = (imgs_np * 255.0).astype("uint8")  # to 0–255
#     BT, _, H, W = imgs.shape
#     for i in range(BT):
#         img = imgs[i, 0]

#         # random horizontal flip
#         if torch.rand(1).item() < 0.5:
#             img = cv2.flip(img, 1)

#         # brightness/contrast jitter
#         alpha = float(torch.empty(1).uniform_(0.9, 1.1))
#         beta = float(torch.empty(1).uniform_(-10, 10))
#         img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

#         imgs[i, 0] = img

#     imgs = imgs.astype("float32") / 255.0
#     return imgs


# # ------------- Training / eval + visualizations -------------

# def train_one_epoch(model, loader, optimizer, device, use_augment=True):
#     model.train()
#     total_loss = 0.0
#     for imgs, gazes, _ in loader:
#         B, T, C, H, W = imgs.shape
#         imgs = imgs.view(B * T, C, H, W).numpy()  # to numpy for OpenCV
#         if use_augment:
#             imgs = augment_eyes(imgs)
#         imgs = torch.from_numpy(imgs).to(device)  # (B*T,1,H,W)

#         gazes = gazes.view(B * T, 2).to(device)

#         optimizer.zero_grad()
#         pred = model(imgs)            # (B*T,2)
#         loss = F.mse_loss(pred, gazes)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     return total_loss / len(loader)


# @torch.no_grad()
# def evaluate_and_visualize(model, loader, device, out_dir="vit_viz"):
#     model.eval()
#     total_loss = 0.0
#     all_pred, all_gt = [], []

#     best_err = float("inf")
#     worst_err = -1.0
#     best_img = None
#     worst_img = None

#     os.makedirs(out_dir, exist_ok=True)

#     for imgs, gazes, _ in loader:
#         B, T, C, H, W = imgs.shape
#         imgs_flat = imgs.view(B * T, C, H, W).to(device)
#         gazes_flat = gazes.view(B * T, 2).to(device)

#         pred = model(imgs_flat)                 # (B*T,2)
#         loss = F.mse_loss(pred, gazes_flat)
#         total_loss += loss.item()

#         pred_cpu = pred.cpu()
#         gt_cpu = gazes_flat.cpu()

#         # angular error per frame
#         err = []
#         for i in range(pred_cpu.shape[0]):
#             e = mean_angular_error_deg(
#                 pred_cpu[i:i+1, :],
#                 gt_cpu[i:i+1, :]
#             )
#             err.append(e)
#         err = torch.tensor(err)

#         imgs_np = imgs_flat.cpu().numpy()  # (B*T,1,H,W)
#         for i in range(err.shape[0]):
#             e = err[i].item()
#             img = imgs_np[i, 0]
#             img_uint8 = (img * 255).clip(0, 255).astype("uint8")

#             if e < best_err:
#                 best_err = e
#                 best_img = img_uint8
#             if e > worst_err:
#                 worst_err = e
#                 worst_img = img_uint8

#         all_pred.append(pred_cpu)
#         all_gt.append(gt_cpu)

#     all_pred = torch.cat(all_pred, dim=0)
#     all_gt = torch.cat(all_gt, dim=0)
#     mae_deg = mean_angular_error_deg(all_pred, all_gt)

#     if best_img is not None:
#         cv2.imwrite(os.path.join(out_dir, f"best_eye_{best_err:.2f}deg.png"),
#                     best_img)
#     if worst_img is not None:
#         cv2.imwrite(os.path.join(out_dir, f"worst_eye_{worst_err:.2f}deg.png"),
#                     worst_img)

#     avg_loss = total_loss / len(loader)
#     return avg_loss, mae_deg, best_err, worst_err


# def main():
#     root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

#     train_ds = EVEEyeSequenceDataset(
#         root=root,
#         split="train",
#         camera="basler",
#         which_eye="left",
#         img_size=(64, 64),
#         seq_len=30,
#         step_stride=30,
#         max_steps=None
#     )
#     val_ds = EVEEyeSequenceDataset(
#         root=root,
#         split="val",
#         camera="basler",
#         which_eye="left",
#         img_size=(64, 64),
#         seq_len=30,
#         step_stride=30,
#         max_steps=None
#     )

#     train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     print("Device:", device)

#     model = ViTEyeGaze(
#         img_size=64,
#         patch_size=8,
#         in_channels=1,
#         embed_dim=192,
#         depth=6,
#         num_heads=6,
#         mlp_dim=384,
#         drop=0.1,
#     ).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)

#     num_epochs = 20
#     for epoch in range(1, num_epochs + 1):
#         tr_loss = train_one_epoch(model, train_loader, optimizer, device, use_augment=True)
#         val_loss, val_mae, best_err, worst_err = evaluate_and_visualize(
#             model, val_loader, device, out_dir="vit_viz"
#         )
#         print(
#             f"[ViT-EyeGaze] Epoch {epoch}/{num_epochs} "
#             f"train_MSE={tr_loss:.4f} | "
#             f"val_MSE={val_loss:.4f} val_MAE={val_mae:.2f} deg "
#             f"(best={best_err:.2f} deg, worst={worst_err:.2f} deg)"
#         )

#     os.makedirs("checkpoints", exist_ok=True)
#     torch.save({"model_state_dict": model.state_dict()},
#                "checkpoints/vit_eyegaze_bigger.pth")


# if __name__ == "__main__":
#     main()


import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_gru_temp import EVEEyeSequenceDataset, mean_angular_error_deg


# ------------- CNN + ViT blocks -------------

class CNNBackbone(nn.Module):
    """
    Small CNN to extract local features from 64x64 eye image.
    Output: (B, C, H', W') with reduced spatial size.
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 64 -> 32
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32 -> 16
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)            # (B,64,16,16)


class PatchEmbeddingFromFeature(nn.Module):
    """
    Patch embedding on top of CNN feature map.
    Fmap size assumed 16x16, patch_size=4 -> 4x4 = 16 patches.
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
            stride=patch_size
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


class CNNEyeViT(nn.Module):
    """
    Hybrid CNN + ViT for single-frame eye gaze (yaw,pitch).
    """
    def __init__(
        self,
        img_size=64,
        in_channels=1,
        embed_dim=192,
        depth=4,
        num_heads=6,
        mlp_dim=384,
        drop=0.1,
    ):
        super().__init__()
        assert img_size == 64
        self.cnn = CNNBackbone(in_channels=in_channels, out_channels=64)
        self.patch_embed = PatchEmbeddingFromFeature(
            fmap_size=16, patch_size=4, in_channels=64, embed_dim=embed_dim
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
        self.head = nn.Linear(embed_dim, 2)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B,1,64,64)
        B = x.shape[0]
        fmap = self.cnn(x)                  # (B,64,16,16)
        tokens = self.patch_embed(fmap)     # (B,N,D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1) # (B,1+N,D)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        pred = self.head(cls_out)
        return pred


# ------------- simple eye augmentation -------------

def augment_eyes(imgs_np):
    imgs = (imgs_np * 255.0).astype("uint8")
    BT, _, H, W = imgs.shape
    for i in range(BT):
        img = imgs[i, 0]
        if torch.rand(1).item() < 0.5:
            img = cv2.flip(img, 1)
        alpha = float(torch.empty(1).uniform_(0.9, 1.1))
        beta = float(torch.empty(1).uniform_(-10, 10))
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        imgs[i, 0] = img
    imgs = imgs.astype("float32") / 255.0
    return imgs


# ------------- Training / eval + visualizations -------------

def train_one_epoch(model, loader, optimizer, device, use_augment=True):
    model.train()
    total_loss = 0.0
    for imgs, gazes, _ in loader:
        B, T, C, H, W = imgs.shape
        imgs = imgs.view(B * T, C, H, W).numpy()
        if use_augment:
            imgs = augment_eyes(imgs)
        imgs = torch.from_numpy(imgs).to(device)
        gazes = gazes.view(B * T, 2).to(device)

        optimizer.zero_grad()
        pred = model(imgs)
        loss = F.mse_loss(pred, gazes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_and_visualize(model, loader, device, out_dir="vit_cnn_viz"):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []
    best_err = float("inf")
    worst_err = -1.0
    best_img = None
    worst_img = None

    os.makedirs(out_dir, exist_ok=True)

    for imgs, gazes, _ in loader:
        B, T, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * T, C, H, W).to(device)
        gazes_flat = gazes.view(B * T, 2).to(device)

        pred = model(imgs_flat)
        loss = F.mse_loss(pred, gazes_flat)
        total_loss += loss.item()

        pred_cpu = pred.cpu()
        gt_cpu = gazes_flat.cpu()

        # per‑frame angular error
        err = []
        for i in range(pred_cpu.shape[0]):
            e = mean_angular_error_deg(
                pred_cpu[i:i+1, :],
                gt_cpu[i:i+1, :]
            )
            err.append(e)
        err = torch.tensor(err)

        imgs_np = imgs_flat.cpu().numpy()
        for i in range(err.shape[0]):
            e = err[i].item()
            img = imgs_np[i, 0]
            img_uint8 = (img * 255).clip(0, 255).astype("uint8")
            if e < best_err:
                best_err = e
                best_img = img_uint8
            if e > worst_err:
                worst_err = e
                worst_img = img_uint8

        all_pred.append(pred_cpu)
        all_gt.append(gt_cpu)

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    mae_deg = mean_angular_error_deg(all_pred, all_gt)

    if best_img is not None:
        cv2.imwrite(os.path.join(out_dir, f"best_eye_{best_err:.2f}deg.png"),
                    best_img)
    if worst_img is not None:
        cv2.imwrite(os.path.join(out_dir, f"worst_eye_{worst_err:.2f}deg.png"),
                    worst_img)

    avg_loss = total_loss / len(loader)
    return avg_loss, mae_deg, best_err, worst_err


def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"

    train_ds = EVEEyeSequenceDataset(
        root=root,
        split="train",
        camera="basler",
        which_eye="left",
        img_size=(64, 64),
        seq_len=30,
        step_stride=30,
        max_steps=None
    )
    val_ds = EVEEyeSequenceDataset(
        root=root,
        split="val",
        camera="basler",
        which_eye="left",
        img_size=(64, 64),
        seq_len=30,
        step_stride=30,
        max_steps=None
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = CNNEyeViT(
        img_size=64,
        in_channels=1,
        embed_dim=192,
        depth=4,
        num_heads=6,
        mlp_dim=384,
        drop=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, use_augment=True)
        val_loss, val_mae, best_err, worst_err = evaluate_and_visualize(
            model, val_loader, device, out_dir="vit_cnn_viz"
        )
        print(
            f"[CNN+ViT-EyeGaze] Epoch {epoch}/{num_epochs} "
            f"train_MSE={tr_loss:.4f} | "
            f"val_MSE={val_loss:.4f} val_MAE={val_mae:.2f} deg "
            f"(best={best_err:.2f} deg, worst={worst_err:.2f} deg)"
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()},
               "checkpoints/vit_cnn_eyegaze.pth")


if __name__ == "__main__":
    main()