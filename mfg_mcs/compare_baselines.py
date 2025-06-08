# compare_baselines.py
import numpy as np
import matplotlib.pyplot as plt
import os

# CONFIG
method_dirs = {
    "main": "results_b3_{b3}",
    "stackelberg": "results_stackelberg_b3_{b3}",
    "auction": "results_auction_fpk_b3_{b3}"
}
labels = {
    "main": "Proposed Method (MFG)",
    "stackelberg": "Stackelberg Baseline",
    "auction": "Auction Baseline"
}

colors = {
    "main": "#1f77b4",
    "stackelberg": "#ff7f0e",
    "auction": "#2ca02c"
}

# 请根据你的路径结构修改以下参数
base_dir = "."
beta3_values = [0.1, 1, 3]
d0 = 1 # 手动切换1/5

for b3 in beta3_values:
    plt.figure(figsize=(7, 4))
    for method, dir_pattern in method_dirs.items():
        dir_path = os.path.join(base_dir, dir_pattern.format(b3=b3))
        avg_rep_file = os.path.join(dir_path, "avg_reputation.npy")
        if not os.path.exists(avg_rep_file):
            print(f"[!] Missing: {avg_rep_file}")
            continue
        avg_rep = np.load(avg_rep_file)
        plt.plot(avg_rep, label=labels[method], linewidth=2.0, color=colors[method])

    plt.xlabel("Time Step")
    plt.ylabel("Average Reputation")
    plt.title(f"Average Reputation Comparison (b3={b3}, d0={d0})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_name = f"compare_avg_rep_b3{b3}_d0{d0}.png"
    plt.savefig(out_name, dpi=300)
    print(f"[✓] Saved: {out_name}")
    plt.close()
