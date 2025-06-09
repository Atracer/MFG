# 改进baseline methods后 比较avg reputation
# 参数为 b1=2 b2=1 b3=1 b4=2 d=1 a1=0.1 a2=1 a3=1 T=50 M=2

import numpy as np
import matplotlib.pyplot as plt
import os

methods = {
    "MFG": "results_d0_1_bench/avg_reputation.npy",
    "Stackelberg": "results_stackelberg_fpk_b3_1/avg_reputation_stackelberg.npy",
    "Auction": "results_auction_fpk_b3_1/avg_reputation.npy",
}

plt.figure(figsize=(8, 5))

for label, path in methods.items():
    if os.path.exists(path):
        data = np.load(path)
        plt.plot(data, label=label)

plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Average Reputation", fontsize=12)
plt.title("avg_reputation", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_avg_reputation.png", dpi=300)
plt.show()

