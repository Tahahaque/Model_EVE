import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_gazerefine_yawpitch import GazeRefineYawPitchDataset
from model_gazerefine_yawpitch_gru import GazeRefineYawPitchGRU
from model_gru_temp import mean_angular_error_deg  # reuse your helper

print("Script imported OK")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_loaders(root, seq_len=30, batch_size=8):
    train_ds = GazeRefineYawPitchDataset(
        root=root,
        split="train01",
        seq_len=seq_len,
        step_stride=seq_len,
        augment=True
    )
    val_ds = GazeRefineYawPitchDataset(
        root=root,
        split="val01",
        seq_len=seq_len,
        step_stride=seq_len,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    n_batches = 0
    for batch in loader:
        screen = batch["screen_seq"].to(DEVICE)        # (B,T,3,72,128)
        eyenet = batch["eyenet_seq"].to(DEVICE)        # (B,T,2)
        gt = batch["gt_yawpitch_seq"].to(DEVICE)       # (B,T,2)

        pred = model(screen, eyenet)                   # (B,T,2)
        loss = criterion(pred, gt)

        total_loss += loss.item()
        # angular error over all time steps
        total_ang += mean_angular_error_deg(
            pred.reshape(-1, 2).cpu(),
            gt.reshape(-1, 2).cpu()
        )
        n_batches += 1

    return total_loss / n_batches, total_ang / n_batches

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
    seq_len = 30
    batch_size = 8
    epochs = 10

    train_loader, val_loader = make_loaders(root, seq_len, batch_size)

    model = GazeRefineYawPitchGRU(
        cnn_out_channels=64,
        gru_hidden_size=128,
        gru_layers=1,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val = 1e9
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            screen = batch["screen_seq"].to(DEVICE)
            eyenet = batch["eyenet_seq"].to(DEVICE)
            gt = batch["gt_yawpitch_seq"].to(DEVICE)

            optimizer.zero_grad()
            pred = model(screen, eyenet)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

        val_loss, val_ang = evaluate(model, val_loader, criterion)
        print(f"[GazeRefine-GRU] Epoch {epoch}/{epochs} "
              f"val_MSE={val_loss:.4f} val_ang={val_ang:.2f} deg")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "val_loss": val_loss,
                 "val_ang": val_ang},
                "checkpoints/gazerefine_yawpitch_gru_best.pth",
            )

if __name__ == "__main__":
    print("Entered main()")
    main()
