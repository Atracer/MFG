'''Plots of Rethink DCT'''
'''Sep 2022'''

# THIS IS FOR PLOTTING
import matplotlib.pyplot as plt  # side-stepping mpl backend

# Other lib
import numpy as np
import settings

# Import results
PATH = 'results_d0_5/'

'''optimal control: single'''
plt.figure(figsize=(12, 9))
x = np.linspace(0, 1, settings.n+1)
opt_u = np.load(PATH + f"/control.npy")

for reward in range(0, settings.G_n+1):
    if reward % 5 == 0:
        plt.plot(x, opt_u[reward, 0, :, 30], linestyle='-', marker='o', markersize=reward*0.5, label='r=%s'%reward)
        #ax1.plot(x, opt_u[reward, 1, :, 30], linestyle='-', marker='o', markersize=reward*0.5, label='g=%s'%reward)

plt.legend(fontsize=15)
plt.xlabel('Reputation', fontsize=20)
plt.ylabel('Data quality', fontsize=20)
plt.title("Population 1's optimal data quality", fontsize=20)

file = "u_reward_single.eps"
plt.savefig(PATH+file, format='eps')
plt.show()

'''0621 optimal control: reward, time'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

x = np.linspace(0, 1, settings.n+1)
opt_u = np.load(PATH + f"/control.npy")
reward1 = 10
reward2 = 15
for time in range(1, settings.time_steps+1):
    if time % 10 == 0 and time !=50:
        ax2.plot(x, opt_u[reward1, 0, :, time], linestyle='-', marker='o', markersize=time*0.1, label='t=%s'%time)
        ax1.plot(x, opt_u[reward1, 1, :, time], linestyle='-', marker='o', markersize=time*0.1, label='t=%s'%time)

ax1.legend(fontsize=15)
ax1.set_xlabel('Reputation', fontsize=20)
ax1.set_ylabel('Data quality', fontsize=15)
ax1.set_title("Population 1", fontsize=20)
ax1.grid()

ax2.legend(fontsize=15)
ax2.set_xlabel('Reputation', fontsize=20)
ax2.set_ylabel('Data quality', fontsize=15)
ax2.set_title("Population 2", fontsize=20)
ax2.grid()
file = "u_reward_time.eps"
plt.savefig(PATH+file, format='eps')
plt.show()

'''FPK 2-D'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
x = np.linspace(0, settings.R, settings.n+1)

opt_delta = np.load(PATH + f"/delta.npy")
reward1 = 0
reward2 = 10
reward3 = 15
l1=ax1.plot(x, opt_delta[reward2, 0, :, 0], marker='o', markersize=2, label='t = 0')
l3=ax1.plot(x, opt_delta[reward2, 0, :, 25], marker='o', markersize=4, label='t = 25')
l5=ax1.plot(x, opt_delta[reward2, 0, :, 50], marker='o', markersize=6, label='t = 50')
ax1.set_xlabel("Reputation (Population1, r = 10)", fontsize=10)
ax1.set_ylabel("PDF", fontsize=20)

l2=ax2.plot(x, opt_delta[reward2, 1, :, 0], marker='o', markersize=2)
l4=ax2.plot(x, opt_delta[reward2, 1, :, 25], marker='o', markersize=4)
l6=ax2.plot(x, opt_delta[reward2, 1, :, 50], marker='o', markersize=6)

ax2.set_xlabel("Reputation (Population2, r = 10)", fontsize=10)


fig.legend(loc='upper center', ncol=15)

file = "fpk_2d_evlo.eps"
plt.savefig(PATH+file, format='eps')

plt.show()

'''FPK 2-D: time - probability'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
x = np.linspace(0, 1, settings.time_steps+1)

opt_delta = np.load(PATH + f"/delta.npy")
reward1 = 0
reward2 = 10
reward3 = 15
pop = 0
ax1.plot(x, opt_delta[reward1, pop, 1, :], marker='o', markersize=2, label='Risk =0.1')
ax1.plot(x, opt_delta[reward1, pop, 15, :], marker='o', markersize=4, label='Risk =0.5')
ax1.plot(x, opt_delta[reward1, pop, 30, :], marker='o', markersize=6, label='Risk =1')
ax1.set_xlabel("Time (r=0)", fontsize=20)
ax1.set_ylabel("PDF", fontsize=20)

l1=ax2.plot(x, opt_delta[reward3, pop, 2, :], marker='o', markersize=2, label='Risk =1')
l3=ax2.plot(x, opt_delta[reward3, pop, 15, :], marker='o', markersize=4, label='Risk =15')
l5=ax2.plot(x, opt_delta[reward3, pop, 30, :], marker='o', markersize=6, label='Risk =30')
ax2.set_xlabel("Time (r=15)", fontsize=20)

labels = ["s=0.1", "s=0.5", "s=1"]

fig.legend([l1, l3, l5], labels=labels, loc='upper center', ncol=15)

file = ("fpk_sir_%s.eps" % pop)
plt.savefig(PATH+file, format='eps')

plt.show()


'''2023-0615: entropy evolution (sub figures)'''
PATH2 = 'results_pop_relation/'
PATH3 = 'results_reward/'
delta1 = np.load(PATH + f"/delta.npy")

entropy1 = np.zeros((settings.G_n+1, settings.time_steps+1))
temp1 = np.zeros((settings.G_n+1, settings.time_steps+1))
x = np.linspace(0, 1, settings.time_steps+1)

for t in range(0, settings.time_steps+1):
    for reward in range(0, settings.G_n+1):
        temp1 = delta1[reward, 0, :, t]
        temp1 /= np.sum(temp1)
        entropy1[reward, t] = -np.sum(temp1 * np.log2(temp1))


for j in range(0, settings.G_n+1):
    if j % 5 == 0:
        i = j-10
        plt.plot(x, entropy1[j, :], marker='o', markersize=i-5, label='r = %s' %j)  #, label='t = %s'%j
plt.legend(loc='upper left', fontsize=15)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Entropy', fontsize=20)

file = "entropy_evol.eps"
plt.savefig(PATH+file, format='eps')
plt.show()


'''Leader's plots'''
# call function from Leader.py
D = np.load(PATH + f"/D.npy")


'''Collective control effort: compare to opt deter'''
U = np.load(PATH + f"/U.npy")
x = np.linspace(0, 1, settings.G_n+1)
for j in range(0, settings.time_steps):
    if j % 10 == 0:
        plt.plot(x, U[:, j], marker='.', markersize=j/5, label='t = %s' %j)  #, label='t = %s'%j
    '''if j % 4 == 0 and 21 < j < 41:
        ax2.plot(x, U[:, j], marker='.', markersize=j-30, label='t = %s' %j)  #, label='t = %s'%j
    if j % 2 == 0 and 41 < j:
        ax3.plot(x, U[:, j], marker='.', markersize=j-35, label='t = %s' %j)  #, label='t = %s'%j
    '''
plt.legend(fontsize=15)

plt.xlabel('Rewards', fontsize=20)
plt.ylabel('Collective data quality', fontsize=20)

#plt.title("Leader's optimization problem", fontsize=20)
file = "collective_u.eps"
plt.savefig(PATH+file, format='eps')
plt.show()


'''Collective control effort: compare to time'''
U = np.load(PATH + f"/U.npy")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
x = np.linspace(0, 1, settings.time_steps+1)
for j in range(0, settings.G_n+1):
    if j == 0:
        ax1.plot(x, U[j, :], marker='.', markersize=j-10, label='r = %s' %j)  #, label='t = %s'%j
    if j == 10:
        ax2.plot(x, U[j, :], marker='.', markersize=j-30, label='r = %s' %j)  #, label='t = %s'%j
    if j == 20:
        ax3.plot(x, U[j, :], marker='.', markersize=j-35, label='r = %s' %j)  #, label='t = %s'%j

ax1.legend(fontsize=15)
ax2.legend(fontsize=15)
ax3.legend(fontsize=15)

ax1.set_xlabel('Time', fontsize=20)
ax2.set_xlabel('Time', fontsize=20)
ax3.set_xlabel('Time', fontsize=20)

ax1.set_ylabel('Collective data quality', fontsize=15)
#plt.title("Leader's optimization problem", fontsize=20)
file = "collective_u_time.eps"
plt.savefig(PATH+file, format='eps')
plt.show()




