import torch
import torch.nn as nn
import torch.nn.functional as F
print("Loaded model_gazerefine_yawpitch_gru.py")

class GazeRefineYawPitchGRU(nn.Module):
    """
    Input:
      screen_seq: (B,T,3,72,128)
      eyenet_seq: (B,T,2)  [yaw,pitch] from EyeNet-GRU

    Output:
      refined yaw,pitch: (B,T,2)
    """
    def __init__(self,
                 cnn_out_channels=64,
                 gru_hidden_size=128,
                 gru_layers=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 72x128 → 36x64
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 18x32
            nn.Conv2d(32, cnn_out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.cnn_out_channels = cnn_out_channels
        self.input_size = cnn_out_channels + 2

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(gru_hidden_size, 2),
        )

    def forward(self, screen_seq, eyenet_seq):
        B, T, C, H, W = screen_seq.shape
        x = screen_seq.view(B * T, C, H, W)
        x = self.encoder(x)                          # (B*T,C_out,1,1)
        x = x.view(B * T, self.cnn_out_channels)     # (B*T,C_out)

        e = eyenet_seq.view(B * T, 2)
        feats = torch.cat([x, e], dim=1)             # (B*T,C_out+2)
        feats = feats.view(B, T, -1)

        gru_out, _ = self.gru(feats)                 # (B,T,H)
        refined = self.head(gru_out)                 # (B,T,2)
        return refined
