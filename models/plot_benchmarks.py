# plot_benchmarks.py
import csv
import matplotlib.pyplot as plt

models = []
params = []
times = []
fps = []

with open("results/model_benchmark.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        models.append(row["model"])
        params.append(int(row["parameters"]))
        times.append(float(row["avg_seq_time_s"]))
        fps.append(float(row["fps"]))

plt.figure(figsize=(6,4))
plt.bar(models, [p/1e6 for p in params])
plt.ylabel("Parameters (Millions)")
plt.title("Model size")
plt.tight_layout()
plt.savefig("results/model_params.png", dpi=200)
plt.close()

plt.figure(figsize=(6,4))
plt.bar(models, times)
plt.ylabel("Avg seq time (s)")
plt.title("Inference time per sequence")
plt.tight_layout()
plt.savefig("results/model_time.png", dpi=200)
plt.close()

plt.figure(figsize=(6,4))
plt.bar(models, fps)
plt.ylabel("Sequences per second")
plt.title("Inference speed (FPS-equivalent)")
plt.tight_layout()
plt.savefig("results/model_fps.png", dpi=200)
plt.close()