import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kfold_utils import make_kfold_splits_two_trains
from dataset_gazerefine_yawpitch import GazeRefineYawPitchDataset
from model_gazerefine_yawpitch_gru import GazeRefineYawPitchGRU
from model_gru_temp import mean_angular_error_deg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    n_batches = 0
    for batch in loader:
        screen = batch["screen_seq"].to(DEVICE)
        eyenet = batch["eyenet_seq"].to(DEVICE)
        gt = batch["gt_yawpitch_seq"].to(DEVICE)

        pred = model(screen, eyenet)
        loss = criterion(pred, gt)

        total_loss += loss.item()
        total_ang += mean_angular_error_deg(
            pred.reshape(-1, 2).cpu(),
            gt.reshape(-1, 2).cpu()
        )
        n_batches += 1
    return total_loss / n_batches, total_ang / n_batches

def main():
    root = "/Users/tahahaque/Documents/model_eve/eve_dataset_2"
    k = 5
    seq_len = 30
    batch_size = 8
    epochs = 10

    folds = make_kfold_splits_two_trains(root, k=k, seed=42)

    fold_results = []

    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Fold {fold_idx+1}/{k} ===")
        train_steps = fold["train"]
        val_steps = fold["val"]

        train_ds = GazeRefineYawPitchDataset(
            root=root,
            split="train",
            seq_len=seq_len,
            step_stride=seq_len,
            augment=True,
            step_dirs=train_steps,
        )
        val_ds = GazeRefineYawPitchDataset(
            root=root,
            split="val",
            seq_len=seq_len,
            step_stride=seq_len,
            augment=False,
            step_dirs=val_steps,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        model = GazeRefineYawPitchGRU(
            cnn_out_channels=64,
            gru_hidden_size=128,
            gru_layers=1,
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_ang = 1e9

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
            print(f"[Fold {fold_idx+1}] Epoch {epoch}/{epochs} "
                  f"val_MSE={val_loss:.4f} val_ang={val_ang:.2f} deg")

        fold_results.append(val_ang)

    fold_results = torch.tensor(fold_results)
    print("\nK-fold results (angular error deg):", fold_results.tolist())
    print("Mean:", fold_results.mean().item())
    print("Std :", fold_results.std(unbiased=False).item())

if __name__ == "__main__":
    main()