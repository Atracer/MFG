import numpy as np
import matplotlib.pyplot as plt

delta = np.load("results_stackelberg_fpk_b3_1/delta.npy")  # 路径视情况修改

pop, n, T_plus1 = delta.shape
rep_grid = np.linspace(0, 1, n)

# t = 0
plt.plot(rep_grid, delta[0, :, 0], label='t=0')
# t = T
plt.plot(rep_grid, delta[0, :, -1], label='t=T')
plt.title("Delta at t=0 and t=T")
plt.legend()
plt.grid(True)
plt.show()