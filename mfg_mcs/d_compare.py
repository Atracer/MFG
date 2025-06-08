# compare_pdf_final_d0.py
import os
import numpy as np
import matplotlib.pyplot as plt
import settings

# ====================== CONFIG =======================
from scipy.stats import gaussian_kde

method_dirs = {
    "main": "results_d0_{d0}",
    "stackelberg": "results_stackelberg_fpk_d0_{d0}",
    "auction": "results_auction_fpk_d0_{d0}"
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

base_dir = "."
d0_values = [1, 5]
b3 = 1  # 固定 b3，用于图名

# ====================== PDF PLOT ======================
for d0 in d0_values:
    plt.figure(figsize=(7, 4))
    for method, dir_pattern in method_dirs.items():
        dir_path = os.path.join(base_dir, dir_pattern.format(d0=d0))
        delta_path = os.path.join(dir_path, "delta.npy")

        if not os.path.exists(delta_path):
            print(f"[!] Missing: {delta_path}")
            continue

        delta = np.load(delta_path)

        # === 获取最终时刻 delta ===
        if delta.ndim == 3:
            delta_final = delta[:, :, -1].sum(axis=0)
        elif delta.ndim == 2:
            delta_final = delta[:, -1]
        elif delta.ndim == 4:
            delta_final = delta[10, :, :, -1].sum(axis=0)
        else:
            raise ValueError(f"Unsupported delta shape: {delta.shape}")

        rep_grid = settings.rep_grid
        rep_axis = np.array(rep_grid.linspace())

        # Normalize
        pdf = delta_final / delta_final.sum()
        plt.plot(rep_axis, pdf, label=labels[method], linewidth=2.0, color=colors[method])

    plt.xlabel("Reputation")
    plt.ylabel("Probability Density")
    plt.title(f"Final State PDF (d0={d0}, b3={b3})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_name = f"compare_pdf_fpk_d0{d0}_b3{b3}.png"
    plt.savefig(out_name, dpi=300)
    print(f"[✓] Saved: {out_name}")
    plt.close()
