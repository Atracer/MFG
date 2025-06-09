import numpy as np
import matplotlib.pyplot as plt
from settings import rep_grid, pop, time_steps  # 确保 settings.py 可导入

# 目录配置（请修改为实际路径）
orig_path = "results_b3_1/"
stackelberg_path = "results_stackelberg_fpk_b3_1"
auction_path = "results_auction_fpk_b3_1"

# # 加载原始方法数据（含 reward index）
control_orig = np.load(f"{orig_path}/control.npy")  # shape: (G_n+1, pop, n_states, T+1)
delta_orig = np.load(f"{orig_path}/delta.npy")

# 加载 Baseline: Stackelberg
control_stack = np.load(f"{stackelberg_path}/control.npy")  # shape: (pop, n_states, T+1)
delta_stack = np.load(f"{stackelberg_path}/delta.npy")

# 加载 Baseline: Auction
control_auction = np.load(f"{auction_path}/control.npy")    # shape: (pop, n_states, T+1)
delta_auction = np.load(f"{auction_path}/delta.npy")

# 参数
rep_values = rep_grid.linspace()
n_states = rep_grid.n + 1
pop_n = pop + 1
T = time_steps
time_to_plot = 30  # 选定时间点

# 平均质量函数
def compute_avg_u(control, delta, t):
    u = np.zeros(n_states)
    for i in range(n_states):
        weighted_sum = 0.0
        total_mass = 0.0
        for k in range(pop_n):
            weighted_sum += control[k, i, t] * delta[k, i, t]
            total_mass += delta[k, i, t]
        u[i] = weighted_sum / total_mass if total_mass > 1e-8 else 0.0
    return u

# 原方法使用 reward_idx=10
reward_idx = 20
control_o = control_orig[reward_idx]  # shape: (pop, n_states, T+1)
delta_o = delta_orig[reward_idx]
u_orig = compute_avg_u(control_o, delta_o, time_to_plot)

# Baselines
u_stack = compute_avg_u(control_stack, delta_stack, time_to_plot)
u_auction = compute_avg_u(control_auction, delta_auction, time_to_plot)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(rep_values, u_orig, label="Original", marker='o')
plt.plot(rep_values, u_stack, label="Stackelberg", marker='s')
plt.plot(rep_values, u_auction, label="Auction", marker='^')

plt.xlabel("Reputation s")
plt.ylabel("Data Quality u(s, t)")
plt.title(f"Data Quality vs Reputation (t={time_to_plot})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_data_quality_vs_reputation.eps", format='eps')
plt.show()