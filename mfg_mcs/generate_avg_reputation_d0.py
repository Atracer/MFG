import numpy as np

# 参数设置
rep_size = 31  # 与 n = 30 相符 → reputation grid size
rep_levels = np.linspace(0, 1, rep_size)  # 声誉状态从 0 到 1
T = 50  # 时间步，与 settings.time_steps 一致
pop = 2  # population 数量，与 settings.pop + 1 一致

# 加载 MFG 主模型结果（d0=1 的基准版本）
delta_all = np.load("results_b3_0.1/delta.npy")  # shape: [reward_idx, pop, rep, T+1]

# 选择某个 reward index（例如 index=10）
reward_index = 10
delta = delta_all[reward_index]  # shape: [pop, rep, T+1]

# 初始化平均声誉序列
avg_reputation = np.zeros(T + 1)
for t in range(T + 1):
    avg_reputation[t] = np.sum(delta[:, :, t] * rep_levels[np.newaxis, :])

# 保存
np.save("results_d0_1_bench/avg_reputation.npy", avg_reputation)
print("平均声誉已保存至 results_d0_1_bench/avg_reputation.npy")
