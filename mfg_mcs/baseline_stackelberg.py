# baseline_stackelberg.py
import numpy as np
import settings
from grid_manager import ReputationGrid
import os
from FPK_Lax import FPK
import matplotlib.pyplot as plt

# Setup
rep_grid = settings.rep_grid         # ReputationGrid 实例
pop = settings.pop + 1               # 种群数量
T = settings.time_steps              # 总时间步数
n_states = rep_grid.n

output_dir = "results_stackelberg_fpk_b3_0.1_june_8"
os.makedirs(output_dir, exist_ok=True)

# parameters
# 领导者相关参数（对应文章中的 ω1, ω2, τ 等）
omega1 = settings.omega1
omega2 = settings.omega2
tau = settings.tau                   # logistic 函数的陡度超参数
d0 = settings.d0                     # 论文中 avg data quality 或者目标 d̄
a1 = settings.a1                     # α1
a2 = settings.a2                     # α2
a3 = settings.a3                     # α3

# initialize array
# 存储数组
control = np.zeros((pop, rep_grid.size(), T + 1))        # 参与者控制强度 u[k, i, t]
delta = np.zeros((pop, rep_grid.size(), T + 1))          # 声誉分布 δ[k, i, t]
reward_seq = np.zeros(T + 1)                             # 领导者奖励轨迹 r(t)
avg_reputation = np.zeros(T + 1)                         # 平均声誉 R(t)
collective_quality = np.zeros(T + 1)                     # 总体数据质量 D(t)

# 使用 settings.init_delta 作为初始声誉分布
for k in range(pop):
    delta[k, :, 0] = settings.init_delta[k, :].copy()

# 领导者初始奖励（论文公式的简化版本）
# 这里可以直接用 β2 * d0 作为初始值，等同 baseline 之前的实现。
reward = settings.beta2 * d0

# 离散时间步长 dt
dt = 1

# simulation loop 
for t in range(T):
    reward_seq[t] = reward

    # —— 参与者最优数据质量 —— 
    # 理想情况下，参与者在给定奖励 r 时，会使偏导 ∂Li/∂d_i = 0 ⇒ d_i* = (α3/α1)*r
    d_opt = (a3 / a1) * reward

    # —— 计算每个 (k, i) 的“瞬时效用” 并转化为成本 cost，再用 sigmoid 得到控制 u —— 
    for k in range(pop):
        for i in rep_grid.all_states():
            s = i * rep_grid.dx
            delta_val = delta[k, i, t]
            # 1) 计算此刻效用 Li(d_opt, s, δ, r) from settings Li 效用函数
            inst_utility = settings.Li(d_opt, delta_val, reward, i)
            # 2) 定义成本 = -InstUtility（效用越大成本越小）
            cost = -inst_utility
            # 3) 通过 logistic(sigmoid) 得到控制强度 [0,1]
            control_value = 1.0 / (1.0 + np.exp(tau * cost))
            control[k, i, t] = control_value

    # —— 构造完整控制张量 u_full 传给 FPK —— 
    u_full = np.zeros((pop, rep_grid.size(), T + 1))
    u_full[:, :, :t+1] = control[:, :, :t+1]

    # —— 调用 FPK 计算下一时刻的声誉分布 —— 
    delta_new = FPK(u_full, reward, it=t)
    delta[:, :, t+1] = delta_new[:, :, t+1]

    # —— 计算平均声誉 R(t) 与整体数据质量 D(t) —— 
    total_rep = 0.0
    total_mass = 0.0
    total_quality = 0.0

    for k in range(pop):
        for i in rep_grid.all_states():
            s = i * rep_grid.dx
            prob = delta[k, i, t]
            total_rep += s * prob
            total_mass += prob
            total_quality += control[k, i, t] * prob

    R = total_rep / total_mass if total_mass > 0 else 0.0
    D = total_quality

    avg_reputation[t] = R
    collective_quality[t] = D

    # —— 领导者动态更新奖励 —— 
    reward += dt * (omega2 * R - omega1 * D)
    reward = max(0.0, reward)

# 保存最后一步的奖励与其他指标
reward_seq[T] = reward
# 再次计算 t = T 时的 R 和 D
total_rep = 0.0
total_mass = 0.0
total_quality = 0.0
for k in range(pop):
    for i in rep_grid.all_states():
        s = i * rep_grid.dx
        prob = delta[k, i, T]
        total_rep += s * prob
        total_mass += prob
        total_quality += control[k, i, T] * prob

avg_reputation[T] = total_rep / total_mass if total_mass > 0 else 0.0
collective_quality[T] = total_quality


# === Visualization: t=0 vs t=T ===
# plt.plot(rep_grid.linspace(), delta[0, :, 0], label='t=0')
# plt.plot(rep_grid.linspace(), delta[0, :, -1], label='t=T')
# plt.title("Delta at t=0 and t=T")
# plt.xlabel("Reputation")
# plt.ylabel("Probability")
# plt.legend()
# plt.grid(True)
# plt.show()


np.save(f"{output_dir}/control.npy", control)
np.save(f"{output_dir}/delta.npy",  delta)
np.save(f"{output_dir}/reward.npy",  reward_seq)
np.save(f"{output_dir}/avg_reputation_stackelberg.npy", avg_reputation)
print("[Stackelberg] Simulation completed.")
print("[DEBUG] delta shape:", delta.shape)
