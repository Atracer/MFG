# baseline_linear.py
import numpy as np
import settings
from grid_manager import ReputationGrid
from HJB_Lax_F import HJB
from FPK_Lax import FPK
import os

# Setup
rep_grid = settings.rep_grid
pop = settings.pop + 1
n = rep_grid.n
T = settings.time_steps
rep_size = rep_grid.size()

# deterministic reward strategy
v_linear = np.linspace(0.2, 0.8, rep_grid.size())

# initial 
control = np.zeros((pop, rep_grid.size(), T + 1))
delta = np.zeros((pop, rep_grid.size(), T + 1))
reward_seq = np.zeros(T + 1)
avg_reputation = np.zeros(T + 1)

# initial distribution
delta[:, :, 0] = settings.init_delta.copy()
control[:, :, 0] = settings.init_u.copy()

# Output directory
output_dir = "results_linear"
os.makedirs(output_dir, exist_ok=True)

# Output arrays
delta_linear = np.zeros((pop, rep_grid.size(), T + 1))
control_linear = np.ones((pop, rep_grid.size(), T + 1))  # Everyone contributes full quality

# Initial distribution
delta_linear[:, :, 0] = settings.init_delta.copy()

# Simulation loop
for t in range(T):
    reward_seq[t] = np.mean(v_linear)

    for k in range(pop):
        u_k = np.zeros(rep_size)

        # 每个状态单独调用 HJB
        for i in rep_grid.all_states():
            # 构造 dummy delta 以匹配 HJB 的三维要求
            delta_for_HJB = np.zeros((pop, rep_size, T + 1))
            delta_for_HJB[k, :, t] = delta[k, :, t]

            u_temp = HJB(control, delta_for_HJB, v_linear[i])
            u_k[i] = u_temp[k, i, t]

        control[k, :, t] = u_k

    # FPK 更新
    delta_new = FPK(control, v_linear, it=1)
    delta[:, :, t + 1] = delta_new[:, :, t + 1]

    # 平均声誉记录
    rep_levels = rep_grid.linspace()
    avg_reputation[t] = np.sum(delta[:, :, t] * rep_levels[np.newaxis, :])

# Save outputs
np.save(f"{output_dir}/control.npy", control)
np.save(f"{output_dir}/delta.npy", delta)
np.save(f"{output_dir}/reward.npy", reward_seq)
np.save(f"{output_dir}/avg_reputation.npy", avg_reputation)

print("[Linear Reward] Simulation completed and saved to results_linear/")
