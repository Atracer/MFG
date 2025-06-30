import numpy as np
import os
import settings
from grid_manager import ReputationGrid
from fpk_lax_revised import FPK

# 输出路径
output_dir = "results_auction_fpk_revised_b3_0.1_june_29"
os.makedirs(output_dir, exist_ok=True)

# 网格与参数
rep_grid: ReputationGrid = settings.rep_grid
pop = settings.pop + 1
T = settings.time_steps
dx = rep_grid.dx

# 参数读取
omega1 = settings.omega1
omega2 = settings.omega2
d0 = settings.d0
a1 = settings.a1
a3 = settings.a3
tau = settings.tau

# 状态变量
control = np.zeros((pop, rep_grid.size(), T + 1))
delta = np.zeros((pop, rep_grid.size(), T + 1))
reward_seq = np.zeros(T + 1)
avg_reputation = np.zeros(T + 1)
collective_quality = np.zeros(T + 1)

# 初始分布使用主模型 delta
for k in range(pop):
    delta[k, :, 0] = settings.init_delta[k, :].copy()

# initial reward + winners
reward_budget = settings.reward_budget  
num_winners = settings.num_winners 

# === Cost function for auction ===
def compute_cost(rep_idx):
    rep = rep_idx * rep_grid.dx
    return settings.beta2 * rep + settings.beta3 * (rep - settings.d0) ** 2

# 主循环
for t in range(T):
    reward_seq[t] = reward_budget
    bids = []

    for k in range(pop):
        for i in rep_grid.all_states():
            cost = compute_cost(i)
            bids.append((cost, k, i))

    # Sort ascending: lower cost is better
    bids.sort(key=lambda x: x[0])
    winners = bids[:num_winners]
    losers = bids[num_winners:]

    u_full = np.zeros((pop, rep_grid.size(), T + 1))

    for cost, k, i in winners:
        # Winner: full control with tiny noise for numeric smoothness
        u_full[k, i, t] = 1.0 + np.random.normal(0, 0.01)
        control[k, i, t] = u_full[k, i, t]

    for cost, k, i in losers:
        # Loser: no control
        u_full[k, i, t] = 0.0
        control[k, i, t] = 0.0

    # === Solve FPK with improved stabilizer ===
    delta_new = FPK(u_full, reward_budget, it=t)
    # [Stability] Clip negative, renormalize, optional smooth
    for k in range(pop):
        delta_new[k, :, t + 1] = np.maximum(delta_new[k, :, t + 1], 0)
        sum_k = np.sum(delta_new[k, :, t + 1]) + 1e-8
        delta_new[k, :, t + 1] /= sum_k

    delta[:, :, t + 1] = delta_new[:, :, t + 1]

    # === Calculate avg reputation ===
    total_rep, total_mass = 0.0, 0.0
    for k in range(pop):
        for i in rep_grid.all_states():
            s = i * rep_grid.dx
            prob = delta[k, i, t]
            total_rep += s * prob
            total_mass += prob

    avg_reputation[t] = total_rep / total_mass if total_mass > 0 else 0.0


# 保存结果
np.save(f"{output_dir}/control.npy", control)
np.save(f"{output_dir}/delta.npy", delta)
np.save(f"{output_dir}/reward.npy", reward_seq)
np.save(f"{output_dir}/avg_reputation.npy", avg_reputation)
np.save(f"{output_dir}/collective_quality.npy", collective_quality)