# import os
# import matplotlib.pyplot as plt

# os.makedirs("results", exist_ok=True)

# epochs = list(range(1, 11))

# # ----- Fill in val_MSE and val_ang for each fold -----
# fold1_MSE = [0.0231, 0.0199, 0.0198, 0.0194, 0.0182,
#              0.0191, 0.0179, 0.0206, 0.0192, 0.0184]
# fold1_ang = [10.66, 9.35, 9.28, 9.20, 9.07,
#              9.09, 8.91, 9.54, 9.16, 8.88]

# fold2_MSE = [0.0164, 0.0139, 0.0134, 0.0126, 0.0120,
#              0.0120, 0.0121, 0.0118, 0.0115, 0.0114]
# fold2_ang = [8.82, 8.17, 7.89, 7.60, 7.25,
#              7.33, 7.38, 7.23, 7.09, 7.09]

# fold3_MSE = [0.0129, 0.0120, 0.0115, 0.0116, 0.0113,
#              0.0108, 0.0104, 0.0104, 0.0106, 0.0103]
# fold3_ang = [7.81, 7.32, 7.00, 7.28, 7.11,
#              6.85, 6.52, 6.78, 6.86, 6.62]

# fold4_MSE = [0.0218, 0.0134, 0.0101, 0.0085, 0.0081,
#              0.0080, 0.0081, 0.0078, 0.0076, 0.0087]
# fold4_ang = [9.90, 8.20, 6.97, 6.00, 5.81,
#              5.65, 5.69, 4.98, 5.37, 5.65]

# fold5_MSE = [0.0190, 0.0148, 0.0149, 0.0146, 0.0148,
#              0.0144, 0.0134, 0.0127, 0.0127, 0.0116]
# fold5_ang = [9.63, 8.15, 8.22, 8.22, 8.33,
#              8.06, 7.71, 7.37, 7.48, 7.30]

# folds_MSE = [fold1_MSE, fold2_MSE, fold3_MSE, fold4_MSE, fold5_MSE]
# folds_ang = [fold1_ang, fold2_ang, fold3_ang, fold4_ang, fold5_ang]

# # final per-fold angular errors from your log:
# final_ang = [8.879860877990723, 7.085959434509277,
#              6.62227725982666, 5.65464973449707,
#              7.302575588226318]

# # ----- 1) Per-fold learning curves (overlayed) -----
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# for i, (mse, ang) in enumerate(zip(folds_MSE, folds_ang), start=1):
#     label = f"Fold {i}"
#     axes[0].plot(epochs, mse, marker="o", label=label)
#     axes[1].plot(epochs, ang, marker="o", label=label)

# axes[0].set_xlabel("Epoch")
# axes[0].set_ylabel("Val MSE")
# axes[0].set_title("GazeRefine-GRU K-fold: val MSE")
# axes[0].grid(True, alpha=0.3)
# axes[0].legend()

# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Val mean angular error (deg)")
# axes[1].set_title("GazeRefine-GRU K-fold: val angular error")
# axes[1].grid(True, alpha=0.3)
# axes[1].legend()

# fig.tight_layout()
# fig.savefig("results/gazerefine_kfold_curves.png", dpi=200)
# plt.close(fig)

# # ----- 2) Final per-fold angular error with mean/std -----
# import numpy as np

# final_ang_arr = np.array(final_ang)
# mean_ang = final_ang_arr.mean()
# std_ang = final_ang_arr.std(ddof=0)  # population std, matches your log

# fold_idx = range(1, 6)

# plt.figure(figsize=(6, 4))
# plt.bar(fold_idx, final_ang_arr, color="tab:blue")
# plt.axhline(mean_ang, color="tab:red", linestyle="--",
#             label=f"Mean = {mean_ang:.2f}°")
# plt.fill_between(
#     [0.5, 5.5],
#     mean_ang - std_ang,
#     mean_ang + std_ang,
#     color="tab:red",
#     alpha=0.1,
#     label=f"±1 std = {std_ang:.2f}°",
# )
# plt.xticks(fold_idx, [f"Fold {i}" for i in fold_idx])
# plt.ylabel("Final val mean angular error (deg)")
# plt.title("GazeRefine-GRU: K-fold final performance")
# plt.legend()
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig("results/gazerefine_kfold_summary.png", dpi=200)
# plt.close()

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)

epochs = list(range(1, 11))

# ----- FOLD 1–4 METRICS ONLY -----
fold1_MSE = [0.0231, 0.0199, 0.0198, 0.0194, 0.0182,
             0.0191, 0.0179, 0.0206, 0.0192, 0.0184]
fold1_ang = [10.66, 9.35, 9.28, 9.20, 9.07,
             9.09, 8.91, 9.54, 9.16, 8.88]

fold2_MSE = [0.0164, 0.0139, 0.0134, 0.0126, 0.0120,
             0.0120, 0.0121, 0.0118, 0.0115, 0.0114]
fold2_ang = [8.82, 8.17, 7.89, 7.60, 7.25,
             7.33, 7.38, 7.23, 7.09, 7.09]

fold3_MSE = [0.0129, 0.0120, 0.0115, 0.0116, 0.0113,
             0.0108, 0.0104, 0.0104, 0.0106, 0.0103]
fold3_ang = [7.81, 7.32, 7.00, 7.28, 7.11,
             6.85, 6.52, 6.78, 6.86, 6.62]

fold4_MSE = [0.0218, 0.0134, 0.0101, 0.0085, 0.0081,
             0.0080, 0.0081, 0.0078, 0.0076, 0.0087]
fold4_ang = [9.90, 8.20, 6.97, 6.00, 5.81,
             5.65, 5.69, 4.98, 5.37, 5.65]

folds_MSE = [fold1_MSE, fold2_MSE, fold3_MSE, fold4_MSE]
folds_ang = [fold1_ang, fold2_ang, fold3_ang, fold4_ang]

# final angular errors for folds 1–4 only
final_ang = [
    8.879860877990723,
    7.085959434509277,
    6.62227725982666,
    5.65464973449707,
]

# ----- 1) Per-fold learning curves (folds 1–4) -----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i, (mse, ang) in enumerate(zip(folds_MSE, folds_ang), start=1):
    label = f"Fold {i}"
    axes[0].plot(epochs, mse, marker="o", label=label)
    axes[1].plot(epochs, ang, marker="o", label=label)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Val MSE")
axes[0].set_title("GazeRefine-GRU K-fold: val MSE")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val mean angular error (deg)")
axes[1].set_title("GazeRefine-GRU K-fold: val angular error")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()
fig.savefig("results/gazerefine_kfold_1to4_curves.png", dpi=200)
plt.close(fig)

# ----- 2) Final per-fold angular error summary (1–4) -----
final_ang_arr = np.array(final_ang)
mean_ang = final_ang_arr.mean()
std_ang = final_ang_arr.std(ddof=0)

fold_idx = range(1, 5)

plt.figure(figsize=(6, 4))
plt.bar(fold_idx, final_ang_arr, color="tab:blue")
plt.axhline(mean_ang, color="tab:red", linestyle="--",
            label=f"Mean = {mean_ang:.2f}°")
plt.fill_between(
    [0.5, 4.5],
    mean_ang - std_ang,
    mean_ang + std_ang,
    color="tab:red",
    alpha=0.1,
    label=f"±1 std = {std_ang:.2f}°",
)
plt.xticks(fold_idx, [f"Fold {i}" for i in fold_idx])
plt.ylabel("Final val mean angular error (deg)")
plt.title("GazeRefine-GRU: K-fold final performance")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/gazerefine_kfold_summary.png", dpi=200)
plt.close()