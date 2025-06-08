# plot_pdf_vs_reputation.py
import numpy as np
import matplotlib.pyplot as plt
import settings

# Load results
delta_stack = np.load("results_stackelberg/delta.npy")
delta_auction = np.load("results_auction/delta.npy")
delta_linear = np.load("results_linear/delta.npy")
delta_mfg = np.load("results_d0_5/delta.npy")[10]  # reward index 10

# Grid and time info
rep_grid = settings.rep_grid
r = rep_grid.linspace()
T = settings.time_steps

# Final time slice PDF (sum over populations)
pdf_stack = delta_stack[:, :, T].sum(axis=0)
pdf_auction = delta_auction[:, :, T].sum(axis=0)
pdf_linear = delta_linear[:, :, T].sum(axis=0)
pdf_mfg = delta_mfg[:, :, T].sum(axis=0)

# Normalize PDFs (in case of numerical drift)
pdf_stack /= np.sum(pdf_stack)
pdf_auction /= np.sum(pdf_auction)
pdf_linear /= np.sum(pdf_linear)
pdf_mfg /= np.sum(pdf_mfg)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(r, pdf_mfg, label='MFG (r=10)', linewidth=2)
plt.plot(r, pdf_stack, '--', label='Stackelberg')
plt.plot(r, pdf_auction, '-.', label='Auction')
plt.plot(r, pdf_linear, ':', label='Linear')
plt.xlabel("Reputation")
plt.ylabel("Probability Density")
plt.title("PDF vs. Reputation at Final Time Step")
plt.legend()
plt.grid()
plt.savefig("compare_pdf_reputation.png")
plt.show()

print("PDF vs. reputation plot saved as compare_pdf_reputation.png")
