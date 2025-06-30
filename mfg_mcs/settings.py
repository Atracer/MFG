import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as plt
from grid_manager import ReputationGrid

# constants
a1 = 0.1  # weight of control:0.1: iteration done within 25 episode
a2 = 1  # weight of contact:1
a3 = 1  # weight of gain

'''Dynamic fi parameters'''
beta1 = 2  # weight of pop influence: benchmark: 2 -> 5 not much difference
beta2 = 1  # weight of current state: 1 -> 2 -> 0.5: huge different!!!!
beta3 = 0.1  # weight of data quality: 1 -> 3 -> 0.1: huge different!!!!
d0 = 1  # average data quality: 1 -> 5 : huge different!!!!
beta4 = 2  # weight of reward: 2 -> 1 -> 4:  huge different!!!!


termin = 1  # termination cost
D = 1.2  # weight of probability
beta = 1  # weight of control in the constrain
omi = 1000

'''population related'''
pop = 1  # number of pop

'''Leader related'''
tau = 4.5  # weight of determination: 3 -> 5 -> 0.1 not much different
omega1 = 0.1  # reward dynamic para 0.1 -> 10 -> 0.01 not much different
omega2 = 0.1  # reward dynamic para 1 -> 5 -> 0.1 not much different

'''Leader discretization'''
g = 0
G0 = g + 0.1  # boundary of reward state
Gm = 0.5 + g  # 1 -> 0.5 -> 1.5
G_n = 20   # discrete state
G = np.linspace(G0, Gm, G_n)  # state from 0 to 1 with 10 points
# print(G)
dG = (Gm - G0)/G_n  # step size


'''initiation'''
''' Discretization the step size: x axis'''
# grid parameters
R = 1  # boundary of risk state
n = 30   # discrete state
rep_grid = ReputationGrid(R=R, n=n)
r = rep_grid.linspace() # state from 0 to 1 with 10 points
# r = np.linspace(0, R, n)  # state from 0 to 1 with 10 points
#print('r', r)
# dx = R/n  # step size
dx = rep_grid.dx  # step size

T = 50
h = 0.02
time_steps = T  # 50

I = 25 # total iteration 25
error = 0.1 # error benchmark 0.1

'''row: discrete, colunm: population [i, n]'''
temp1 = np.ones(2*rep_grid.size(),)  # create an array
temp2 = np.ones(2*rep_grid.size(),)  # create an array
temp3 = np.ones(2*rep_grid.size(),)  # create an array
temp4 = np.ones(2*rep_grid.size(),)  # create an array

init_v = np.reshape(temp1, (2, rep_grid.size(),))  # reshape the array into matrix
term_v = np.reshape(temp2, (2, rep_grid.size(),))  # reshape the array into matrix
init_delta = np.reshape(temp3, (2, rep_grid.size(),))
init_u = np.reshape(temp4, (2, rep_grid.size(),))


# value function t=0
v = n * np.ones(r.shape)  # initial value at all states
v[-1] = 20  # r = 1, t = 0, value = 20
v[0] = 1  # r = 0, t = 0, value = 1
init_v[0, :] = v
termv = 30 * np.ones(r.shape)  # t = T, at all state, value function
term_v[0, :] = termv

'''HJB_Lax_F.py'''
v_hjb = np.ones(n+1)

'''Leader.py'''
v0 = np.ones(G_n+1)
D = np.ones(G_n+1)

# probability distribution t = 0
# init_delta[0, :] = 0.85 * np.ones(r.shape)
# init_delta[1, :] = 0.15 * np.ones(r.shape)

# method 2
mu0, sigma0 = 0.3, 0.1  
mu1, sigma1 = 0.7, 0.1 
init_delta[0, :] = norm.pdf(r, loc=mu0, scale=sigma0)
init_delta[1, :] = norm.pdf(r, loc=mu1, scale=sigma1)

init_delta[0, :] /= init_delta[0, :].sum()
init_delta[1, :] /= init_delta[1, :].sum()

'''FPK_Lax_F.py'''
mu = 2
mu1 = 5
init_m = poisson.pmf(np.arange(len(r)+1), mu)
init_m1 = poisson.pmf(np.arange(len(r)+1), mu1)

#print(len(init_m))
#plt.plot(np.arange(10), init_m, init_m1)
#plt.show()

# control initialization t = 0
# assuming control \in (0,1), starting with neutral 0.5
init_u[0, :] = 0.5 * np.ones(r.shape)
init_u[1, :] = 0.5 * np.ones(r.shape)

'''FPK_Lax-F'''
temp_u = np.zeros(n+1)
'''Leader'''
temp_D = np.zeros(G_n+1)
'''discretilize steps'''
ipos = np.zeros(n + 1)
ineg = np.zeros(n + 1)
for i in range(0, n + 1):
    ineg[i] = i - 1
    ipos[i] = i + 1
ipos[n] = 0
ineg[0] = n
'''leader's discretilize steps'''
lipos = np.zeros(G_n + 1)
lineg = np.zeros(G_n + 1)
for i in range(0, G_n + 1):
    lineg[i] = i - 1
    lipos[i] = i + 1
lipos[G_n] = 0
lineg[0] = G_n


iipos = np.zeros(n + 1)
iineg = np.zeros(n + 1)
for i in range(0, n + 1):
    iineg[i] = i - 2
    iipos[i] = i + 2

iipos[n] = 1
iipos[n-1] = 0
iineg[0] = n-1
iineg[1] = n

org = np.zeros(n+1)
for i in range(0, n+1):
    org[i] = i

liipos = np.zeros(G_n + 1)
liineg = np.zeros(G_n + 1)
for i in range(0,  + 1):
    liineg[i] = i - 2
    liipos[i] = i + 2

liipos[G_n] = 1
liipos[G_n-1] = 0
liineg[0] = G_n-1
liineg[1] = G_n

lorg = np.zeros(G_n+1)
for i in range(0, G_n+1):
    lorg[i] = i


#new one should be right
def fi(u, delta, g, i, k):
    #delta = init_m[0]  # pop i's mean field term
    f = 0
    for num in range(0, pop+1):
        if k != num:
            f = beta1 * delta - beta2 * i * dx + beta3 * (i * dx - d0) * u + beta4 * g
    return f

def Li(u, delta, g, i):
    Li = 0.5 * a1 * (u ** 2) + (i * dx + a2 * delta) - a3 * g * u
    return Li

# auction parameters
reward_budget = 1.0
num_winners = 5
