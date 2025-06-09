# baseline_auction.py
import numpy as np
import settings
from grid_manager import ReputationGrid
import os
from FPK_Lax import FPK

# Setup
rep_grid = settings.rep_grid
pop = settings.pop + 1
T = settings.time_steps
r = rep_grid.linspace()
n = rep_grid.n

output_dir = "results_auction_fpk_b3_3"
os.makedirs(output_dir, exist_ok=True)

# parameters
beta2 = settings.beta2
beta3 = settings.beta3
d0 = settings.d0

rep_d0 = d0
reward = beta2 * rep_d0 + beta3 * (rep_d0 - d0)**2
num_winners = settings.num_winners if hasattr(settings, 'num_winners') else 5

# initial states
delta = np.zeros((pop, rep_grid.size(), T + 1))
delta[:, :, 0] = settings.init_delta.copy()

control = np.zeros((pop, rep_grid.size(), T + 1))
reward_seq = np.zeros(T + 1)
avg_reputation = np.zeros(T + 1)
 
# cost function
def compute_cost(rep_idx):
    rep = rep_idx * rep_grid.dx
    # return beta2 * rep_idx * rep_grid.dx + beta3 * (rep - d0)
    return beta2 * rep + beta3 * (rep - d0) ** 2 

# loop 
for t in range(T):
    reward_seq[t] = reward
    total_rep = 0
    total_mass = 0
    bids = []

    for k in range(pop):
        for i in rep_grid.all_states():
            cost = compute_cost(i)
            bids.append((cost, k, i))

    # sort by cost (ascending)
    bids.sort(key=lambda x: x[0])
    bid_values = np.array([b[0] for b in bids])
    norm_scores = 1.0 - (bid_values - bid_values.min()) / (bid_values.max() - bid_values.min() + 1e-8)

    # assign soft score as control probability
    u_full = np.zeros((pop, rep_grid.size(), T + 1))
    for idx, (score, (cost, k, i)) in enumerate(zip(norm_scores, bids)):
        u_full[k, i, t] = score
        control[k, i, t] = score

    delta_new = FPK(u_full, reward, it=t)
    delta[:, :, t + 1] = delta_new[:, :, t + 1]

    for k in range(pop):
        for i in rep_grid.all_states():
            rep = i * rep_grid.dx
            total_rep += rep * delta[k, i, t]
            total_mass += delta[k, i, t]
    avg_reputation[t] = total_rep / total_mass if total_mass > 0 else 0

# Save outputs
np.save(f"{output_dir}/control.npy", control)
np.save(f"{output_dir}/delta.npy", delta)
np.save(os.path.join(output_dir, "reward.npy"), reward_seq)
np.save(os.path.join(output_dir, "avg_reputation.npy"), avg_reputation)
print("[Auction] Simulation completed and saved to results_auction/")
