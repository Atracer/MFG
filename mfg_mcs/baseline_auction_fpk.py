import numpy as np
import os
import settings
from grid_manager import ReputationGrid
from FPK_Lax import FPK

# 输出路径
output_dir = "results_auction_fpk_b3_0.1"
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

# 初始 reward 设置（Eq. 25 起始值）
reward = settings.beta2 * d0
dt = 1

# 主循环
for t in range(T):
    reward_seq[t] = reward
    d_opt = (a3 / a1) * reward  # follower 最优努力

    # 控制策略（soft sigmoid）
    for k in range(pop):
        for i in rep_grid.all_states():
            delta_val = delta[k, i, t]
            cost = -settings.Li(d_opt, delta_val, reward, i)
            control[k, i, t] = 1.0 / (1.0 + np.exp(tau * cost))

    # 状态转移（FPK）
    u_full = np.zeros((pop, rep_grid.size(), T + 1))
    u_full[:, :, :t + 1] = control[:, :, :t + 1]
    delta_new = FPK(u_full, reward, it=t)
    delta[:, :, t + 1] = delta_new[:, :, t + 1]

    # 聚合统计
    total_rep = total_mass = total_quality = 0.0
    for k in range(pop):
        for i in rep_grid.all_states():
            s = i * dx
            prob = delta[k, i, t]
            total_rep += s * prob
            total_mass += prob
            total_quality += control[k, i, t] * prob

    R = total_rep / total_mass if total_mass > 0 else 0.0
    D = total_quality
    avg_reputation[t] = R
    collective_quality[t] = D

    # ★ 奖励更新：Eq. (25) 的优化梯度形式（非 Eq.6）
    reward += dt * (omega2 * R - omega1 * D)
    reward = max(0.0, reward)  # 非负限制

# 最后一时刻统计更新
reward_seq[T] = reward
total_rep = total_mass = total_quality = 0.0
for k in range(pop):
    for i in rep_grid.all_states():
        s = i * dx
        prob = delta[k, i, T]
        total_rep += s * prob
        total_mass += prob
        total_quality += control[k, i, T] * prob
avg_reputation[T] = total_rep / total_mass if total_mass > 0 else 0.0
collective_quality[T] = total_quality

# 保存结果
np.save(f"{output_dir}/control.npy", control)
np.save(f"{output_dir}/delta.npy", delta)
np.save(f"{output_dir}/reward.npy", reward_seq)
np.save(f"{output_dir}/avg_reputation.npy", avg_reputation)
np.save(f"{output_dir}/collective_quality.npy", collective_quality)