import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ 声誉等级设置（默认 n=30） -------------------
rep_size = 31
rep_levels = np.linspace(0, 1, rep_size)
T = 50
pop = 2
final_t = T
reward_index = 10

# ------------------ 结果路径 -------------------
baseline_dirs = {
    "Stackelberg": "results_stackelberg",
    "Auction": "results_auction"
}
mfg_dir = "results_d0_1_bench"

# ------------------ 图1：PDF at t=50 -------------------
plt.figure(figsize=(8, 5))
delta_mfg_all = np.load(os.path.join(mfg_dir, "delta.npy"))
delta_mfg = delta_mfg_all[reward_index]
pdf_mfg = np.mean(delta_mfg[:, :, final_t], axis=0)
plt.plot(rep_levels, pdf_mfg, label="MFG", linewidth=2)

for label, path in baseline_dirs.items():
    delta = np.load(os.path.join(path, "delta.npy"))
    pdf = np.mean(delta[:, :, final_t], axis=0)
    plt.plot(rep_levels, pdf, label=label, linestyle='--')

plt.title("PDF at Final Time Step (t=50)")
plt.xlabel("Reputation")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pdf_final_t50.png")

# ------------------ 图2：Control at t=50 -------------------
plt.figure(figsize=(8, 5))
control_mfg_all = np.load(os.path.join(mfg_dir, "control.npy"))
control_mfg = control_mfg_all[reward_index]
ctrl_avg = np.mean(control_mfg[:, :, final_t], axis=0)
plt.plot(rep_levels, ctrl_avg, label="MFG", linewidth=2)

for label, path in baseline_dirs.items():
    ctrl = np.load(os.path.join(path, "control.npy"))
    ctrl_avg = np.mean(ctrl[:, :, final_t], axis=0)
    plt.plot(rep_levels, ctrl_avg, label=label, linestyle='--')

plt.title("Control Strategy at Final Time Step (t=50)")
plt.xlabel("Reputation")
plt.ylabel("Control Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("control_final_t50.png")

# ------------------ 图3：Reward vs Reputation -------------------
# plt.figure(figsize=(8, 5))
# reward_mfg = np.load(os.path.join(mfg_dir,  "reward.npy"))
# g_t = reward_mfg[final_t]
# reward_curve = g_t * rep_levels
# plt.plot(rep_levels, reward_curve, label="MFG (g*r)", linewidth=2)

# plt.plot(rep_levels, 0.5 * np.ones_like(rep_levels), label="Stackelberg", linestyle='--')
# plt.plot(rep_levels, 0.1 * np.ones_like(rep_levels), label="Auction (approx)", linestyle='--')

# plt.title("Reward vs Reputation at Final Time Step (t=50)")
# plt.xlabel("Reputation")
# plt.ylabel("Reward Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("reward_vs_reputation.png")

print("三张图已保存：pdf_final_t50.png, control_final_t50.png")
