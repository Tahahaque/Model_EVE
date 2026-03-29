import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# 1) Patch embedding: image -> sequence of patch tokens
# -------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects each to an embedding.

    Input:  (B, C, H, W)
    Output: (B, N, D) where N = number of patches, D = embed_dim
    """
    def __init__(self, img_size=224, patch_size=16,
                 in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv with kernel = patch_size and stride = patch_size is a fast
        # way to extract and linearly project patches.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x)              # (B, D, H/P, W/P)
        x = x.flatten(2)              # (B, D, N)
        x = x.transpose(1, 2)         # (B, N, D)
        return x


# -------------------------------------------------
# 2) Transformer encoder block (MHSA + MLP)
# -------------------------------------------------

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
    """
    One ViT encoder block: LayerNorm -> Multi‑Head Self‑Attention -> residual,
    then LayerNorm -> MLP -> residual.
    """
    def __init__(self, dim, num_heads, mlp_dim, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,  # so we use (B, N, D)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, drop=drop)

    def forward(self, x):
        # x: (B, N, D)
        # self‑attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x


# -------------------------------------------------
# 3) Vision Transformer
# -------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Minimal Vision Transformer for image classification.

    Steps:
      1) Split image into patches and embed (PatchEmbedding).
      2) Prepend a learnable [CLS] token.
      3) Add learnable positional embeddings.
      4) Pass through L encoder blocks (self‑attention + MLP).
      5) Use the [CLS] token representation for classification.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_dim=768,
        drop=0.0,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings for [CLS] + all patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop)

        # Stack of Transformer encoder blocks
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

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x: (B, C, H, W) image tensor, H=W=img_size
        Returns logits: (B, num_classes)
        """
        B = x.shape[0]

        # 1) patch embedding -> (B, N, D)
        x = self.patch_embed(x)

        # 2) prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1+N, D)

        # 3) add positional embeddings
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        # 4) transformer encoder blocks
        for blk in self.blocks:
            x = blk(x)

        # 5) classification from [CLS] token
        x = self.norm(x)
        cls_out = x[:, 0]             # (B, D)
        logits = self.head(cls_out)   # (B, num_classes)
        return logits


# -------------------------------------------------
# 4) Tiny usage example
# -------------------------------------------------

if __name__ == "__main__":
    # Example: 32x32 images, 4x4 patches, CIFAR‑10‑like setup
    img_size = 32
    patch_size = 4
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        drop=0.1,
    )

    x = torch.randn(8, 3, img_size, img_size)  # batch of 8 images
    logits = model(x)
    print("logits shape:", logits.shape)  # (8, 10)