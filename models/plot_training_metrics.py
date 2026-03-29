# plot_training_metrics.py
import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# -----------------------------
# 1. Hard-code your metrics
# -----------------------------

epochs_5 = [1, 2, 3, 4, 5]
epochs_10 = list(range(1, 11))

# Camera – MobileNet
cam_mobilenet_train_loss = [0.2033, 0.1030, 0.0953, 0.0897, 0.0897]
cam_mobilenet_val_loss   = [1.8156, 0.2374, 0.5972, 0.2230, 0.4044]
cam_mobilenet_val_ang    = [16.69,  9.68,   11.82,  8.57,   6.26]

# Basler – MobileNet
bas_mobilenet_train_loss = [0.1328, 0.1063, 0.1026, 0.1007, 0.0985]
bas_mobilenet_val_loss   = [0.2173, 0.5759, 0.3906, 0.8907, 0.6954]
bas_mobilenet_val_ang    = [6.31,   6.29,   4.04,   6.29,   6.45]

# Basler – EfficientNet
bas_eff_train_loss = [0.1632, 0.1130, 0.1076, 0.1039, 0.1008]
bas_eff_val_loss   = [0.2908, 0.3821, 0.6723, 0.7212, 0.8690]
bas_eff_val_ang    = [4.21,   6.66,   14.68,  12.85,  13.39]

# Camera – ResNet
cam_resnet_train_loss = [0.1508, 0.1098, 0.1041, 0.1013, 0.0994]
cam_resnet_val_loss   = [0.2178, 0.2166, 0.3616, 0.2234, 0.4329]
cam_resnet_val_ang    = [8.22,  12.58,   7.66,  14.74,  13.21]

# EyeNet-GRU temporal model
gru_train_loss = [0.2321, 0.1678, 0.1635, 0.1482, 0.1501]
gru_val_loss   = [0.8938, 0.2426, 0.1846, 0.1545, 0.1934]
gru_val_ang    = [13.64,  8.68,   9.13,   6.49,   5.41]

# GazeRefine-GRU (10 epochs)
refine_val_MSE = [0.0185, 0.0169, 0.0160, 0.0155, 0.0153,
                  0.0149, 0.0143, 0.0146, 0.0134, 0.0116]
refine_val_ang = [9.39, 8.82, 8.51, 8.17, 8.05,
                  8.02, 7.94, 7.87, 7.53, 7.08]

# -----------------------------
# 2. Helper: plot one model
# -----------------------------

def plot_model_curves(name, epochs, train_loss, val_loss, val_ang, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    axes[0].plot(epochs, train_loss, marker="o", label="Train loss")
    axes[0].plot(epochs, val_loss, marker="s", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{name}: loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Angular error
    axes[1].plot(epochs, val_ang, marker="o", color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val mean angular error (deg)")
    axes[1].set_title(f"{name}: val angular error")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -----------------------------
# 3. Make per-model figures
# -----------------------------

plot_model_curves(
    "Camera – MobileNet",
    epochs_5,
    cam_mobilenet_train_loss,
    cam_mobilenet_val_loss,
    cam_mobilenet_val_ang,
    "results/camera_mobilenet_curves.png",
)

plot_model_curves(
    "Basler – MobileNet",
    epochs_5,
    bas_mobilenet_train_loss,
    bas_mobilenet_val_loss,
    bas_mobilenet_val_ang,
    "results/basler_mobilenet_curves.png",
)

plot_model_curves(
    "Basler – EfficientNet",
    epochs_5,
    bas_eff_train_loss,
    bas_eff_val_loss,
    bas_eff_val_ang,
    "results/basler_efficientnet_curves.png",
)

plot_model_curves(
    "Camera – ResNet",
    epochs_5,
    cam_resnet_train_loss,
    cam_resnet_val_loss,
    cam_resnet_val_ang,
    "results/camera_resnet_curves.png",
)

plot_model_curves(
    "EyeNet-GRU",
    epochs_5,
    gru_train_loss,
    gru_val_loss,
    gru_val_ang,
    "results/eyenet_gru_curves.png",
)

# Refine: only val metrics available
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs_10, refine_val_MSE, marker="o")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Val MSE")
axes[0].set_title("GazeRefine-GRU: val MSE")
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_10, refine_val_ang, marker="o", color="tab:red")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val mean angular error (deg)")
axes[1].set_title("GazeRefine-GRU: val angular error")
axes[1].grid(True, alpha=0.3)

fig.suptitle("GazeRefine-GRU")
fig.tight_layout()
fig.savefig("results/gazerefine_gru_curves.png", dpi=200)
plt.close(fig)

# -----------------------------
# 4. Final-epoch comparison figure
# -----------------------------

models = [
    "Cam-MobileNet",
    "Basler-MobileNet",
    "Basler-EfficientNet",
    "Cam-ResNet",
    "EyeNet-GRU",
    "GazeRefine-GRU",
]

final_val_ang = [
    cam_mobilenet_val_ang[-1],
    bas_mobilenet_val_ang[-1],
    bas_eff_val_ang[-1],
    cam_resnet_val_ang[-1],
    gru_val_ang[-1],
    refine_val_ang[-1],
]

plt.figure(figsize=(8, 4))
xs = range(len(models))
plt.bar(xs, final_val_ang, color="tab:blue")
plt.xticks(xs, models, rotation=30, ha="right")
plt.ylabel("Final val mean angular error (deg)")
plt.title("Cross-model comparison of final validation error")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/final_val_error_comparison.png", dpi=200)
plt.close()