# compare_pdf_final.py
import os
import numpy as np
import matplotlib.pyplot as plt
import settings

# ====================== CONFIG =======================
from scipy.stats import gaussian_kde

method_dirs = {
    "main": "results_b3_{b3}",
    "stackelberg": "results_stackelberg_fpk_b3_{b3}",
    # "stk":"results_stackelberg_fpk_b3_{b3}",
    "auction": "results_auction_fpk_b3_{b3}"
}

labels = {
    "main": "Proposed Method (MFG)",
    "stackelberg": "Stackelberg Baseline",
    # "stk":"stk",
    "auction": "Auction Baseline"
}

colors = {
    "main": "#1f77b4",
    "stackelberg": "#ff7f0e",
    # "stk":"#b41f1f",
    "auction": "#2ca02c"
}

base_dir = "."
beta3_values = [0.1, 1, 3]
d0 = 1  # 默认 d0 参数名

# ====================== PDF PLOT ======================
for b3 in beta3_values:
    plt.figure(figsize=(7, 4))
    for method, dir_pattern in method_dirs.items():
        dir_path = os.path.join(base_dir, dir_pattern.format(b3=b3))
        delta_path = os.path.join(dir_path, "delta.npy")

        if not os.path.exists(delta_path):
            print(f"[!] Missing: {delta_path}")
            continue

        delta = np.load(delta_path)

        # === 获取最终时刻 delta ===
        if delta.ndim == 3:
            # shape: (pop, rep, time)
            delta_final = delta[:, :, -1].sum(axis=0)
        elif delta.ndim == 2:
            delta_final = delta[:, -1]
        elif delta.ndim == 4:
            # shape: (reward_idx, pop, rep, time)
            delta_final = delta[10, :, :, -1].sum(axis=0)
            print(f"[DEBUG] {method} max delta_final = {np.max(delta_final):.4f}, sum = {np.sum(delta_final):.4f}")
        else:
            raise ValueError(f"Unsupported delta shape: {delta.shape}")

        rep_grid = settings.rep_grid
        rep_axis = np.array(rep_grid.linspace())

        # Normalize
        pdf = delta_final / delta_final.sum()
        plt.plot(rep_axis, pdf, label=labels[method], linewidth=2.0, color=colors[method])

    plt.xlabel("Reputation")
    plt.ylabel("Probability Density")
    plt.title(f"Final State PDF (b3={b3}, d0={d0})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_name = f"compare_pdf_fpk_b3{b3}_d0{d0}_new.png"
    plt.savefig(out_name, dpi=300)
    print(f"[✓] Saved: {out_name}")
    plt.close()
