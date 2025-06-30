import numpy as np
import settings
from grid_manager import ReputationGrid
import os
from fpk_lax_revised import FPK
import matplotlib.pyplot as plt

# Setup
rep_grid = settings.rep_grid         # ReputationGrid 实例
pop = settings.pop + 1               # 种群数量
T = settings.time_steps              # 总时间步数
n_states = rep_grid.n

output_dir = "results_stackelberg_fpk_revised_b3_3_june_29"
os.makedirs(output_dir, exist_ok=True)

# parameters
omega1 = settings.omega1
omega2 = settings.omega2
tau = settings.tau
d0 = settings.d0
a1 = settings.a1
a2 = settings.a2
a3 = settings.a3

# initialize array
control = np.zeros((pop, rep_grid.size(), T + 1))
delta = np.zeros((pop, rep_grid.size(), T + 1))
reward_seq = np.zeros(T + 1)
avg_reputation = np.zeros(T + 1)
collective_quality = np.zeros(T + 1)

# 使用 settings.init_delta 作为初始声誉分布
for k in range(pop):
    delta[k, :, 0] = settings.init_delta[k, :].copy()

# === Initial reward: static + noise ===
reward = settings.beta2 * d0

dt = 1

# simulation loop 
for t in range(T):
    reward_seq[t] = reward

    # Approximate optimal data quality (still heuristic)
    d_opt = (a3 / a1) * reward

    for k in range(pop):
        for i in rep_grid.all_states():
            s = i * rep_grid.dx
            delta_val = delta[k, i, t]

            # Instant utility
            inst_utility = settings.Li(d_opt, delta_val, reward, i)

            # Soft control: logistic on negative utility
            cost = -inst_utility
            control_value = 1.0 / (1.0 + np.exp(tau * cost))
            control[k, i, t] = control_value

    # FPK solve
    u_full = np.zeros((pop, rep_grid.size(), T + 1))
    u_full[:, :, :t+1] = control[:, :, :t+1]
    delta_new = FPK(u_full, reward, it=t)
    delta[:, :, t+1] = delta_new[:, :, t+1]

    # Calculate avg reputation & collective data quality
    total_rep, total_mass, total_quality = 0.0, 0.0, 0.0
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

    # === Weak leader feedback: fixed + small noise every 10 steps ===
    if t % 10 == 0:
        reward += np.random.normal(0, 0.05)
        reward = max(0.0, reward)  # keep positive

# Final step save
reward_seq[T] = reward
# Recompute R, D at T
total_rep, total_mass, total_quality = 0.0, 0.0, 0.0
for k in range(pop):
    for i in rep_grid.all_states():
        s = i * rep_grid.dx
        prob = delta[k, i, T]
        total_rep += s * prob
        total_mass += prob
        total_quality += control[k, i, T] * prob

avg_reputation[T] = total_rep / total_mass if total_mass > 0 else 0.0
collective_quality[T] = total_quality

# === Save results ===
np.save(f"{output_dir}/control.npy", control)
np.save(f"{output_dir}/delta.npy", delta)
np.save(f"{output_dir}/reward.npy", reward_seq)
np.save(f"{output_dir}/avg_reputation_stackelberg.npy", avg_reputation)

print("[Stackelberg Baseline Revised] Simulation completed.")
print("[DEBUG] delta shape:", delta.shape)