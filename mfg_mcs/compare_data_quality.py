import numpy as np
import matplotlib.pyplot as plt
import os

# 你的 grid 设置
class ReputationGrid:
    def __init__(self, R=1.0, n=30):
        self.n = n
        self.R = R
        self.dx = R / n
        self.r = np.linspace(0, R, n + 1)

    def linspace(self):
        return self.r

    def size(self):
        return self.n + 1

# 配置
rep_grid = ReputationGrid(R=1.0, n=30)
r = rep_grid.linspace()
d0 = 1  # 根据你的 settings.py 设置
pop = 2
beta3_value = 1  # 指定 beta3

# 比较的三个方法路径
method_dirs = {
    "main": "results_b3_1",
    "stackelberg": "results_stackelberg_fpk_b3_1",
    "auction": "results_auction_fpk_b3_1"
}

colors = {
    "main": "#1f77b4",
    "stackelberg": "#ff7f0e",
    "auction": "#2ca02c"
}

labels = {
    "main": "Proposed Method (MFG)",
    "stackelberg": "Stackelberg Baseline",
    "auction": "Auction Baseline"
}

plt.figure(figsize=(8, 5))

for method, path in method_dirs.items():
    delta_path = os.path.join(path, "delta.npy")
    if not os.path.exists(delta_path):
        print(f"[!] Missing: {delta_path}")
        continue

    delta = np.load(delta_path)

    if delta.ndim == 3:
        delta_final = delta[:, :, -1].sum(axis=0)
    elif delta.ndim == 2:
        delta_final = delta[:, -1]
    elif delta.ndim == 4:
        delta_final = delta[10, :, :, -1].sum(axis=0)
    else:
        print(f"[!] Unrecognized delta shape: {delta.shape}")
        continue

    dq_pointwise = ((r - d0) ** 2) * delta_final

    plt.plot(r, dq_pointwise, label=labels[method], linewidth=2.0, color=colors[method])

plt.xlabel("Reputation")
plt.ylabel("Data Quality Contribution")
plt.title(f"Data Quality vs. Reputation (β₃={beta3_value})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"compare_dq_vs_reputation_b3_{beta3_value}.png", dpi=300)
plt.show()
