# compare_d0_pdf_main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import settings

# ==================== CONFIG ======================
base_dir = "."
method = "main"
d0_values = [1, 5]
b3 = 1

label_map = {
    1: "d0 = 1",
    5: "d0 = 5"
}

colors = {
    1: "#1f77b4",
    5: "#d62728"
}

# ==================== PLOT PDF FOR MAIN ============
plt.figure(figsize=(7, 4))
for d0 in d0_values:
    dir_path = os.path.join(base_dir, f"results_d0_{d0}")
    delta_path = os.path.join(dir_path, "delta.npy")

    if not os.path.exists(delta_path):
        print(f"[!] Missing: {delta_path}")
        continue

    delta = np.load(delta_path)
    if delta.ndim == 4:
        delta_final = delta[10, :, :, -1].sum(axis=0)
    elif delta.ndim == 3:
        delta_final = delta[:, :, -1].sum(axis=0)
    else:
        raise ValueError("Unsupported delta shape")

    rep_axis = np.array(settings.rep_grid.linspace())
    pdf = delta_final / delta_final.sum()
    plt.plot(rep_axis, pdf, label=label_map[d0], linewidth=2.0, color=colors[d0])

plt.xlabel("Reputation")
plt.ylabel("Probability Density")
plt.title("Impact of d0 on Final Reputation Distribution (Main Method)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_d0_pdf_main.png", dpi=300)
print("[✓] Saved: compare_d0_pdf_main.png")
plt.close()
