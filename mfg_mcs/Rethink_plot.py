# This is the main program
# matplotlib inline
import matplotlib.pyplot as plt  # side-stepping mpl backend
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# Other lib
import numpy as np

## import FPK, HJB, Leader
from FPK_Lax import FPK
from HJB_Lax_F import HJB
from settings import Li
from Leader import Leader
import settings
#from Leader import Leader
'''init'''
time_steps = settings.time_steps
N = settings.n
dx = settings.R/N
pop = 1

PATH = 'results_test/'
'''Followers: init control (reward, population, state, time)'''
u = np.zeros((pop + 1, N + 1, time_steps + 1))
opt_control = np.zeros((settings.G_n + 1, pop+1, N+1, time_steps+1))  # (reward, pop, state, time)
opt_delta = np.zeros((settings.G_n + 1, pop+1, N+1, time_steps+1))  # (reward, pop, state, time)
opt_value = np.zeros((settings.G_n + 1, pop+1, N+1, time_steps+1))  # (reward, pop, state, time)

for i in range(0, N + 1):
    u[0, i, 0] = settings.temp_u[i] + 0.1 #(population, state, time)
    u[1, i, 0] = settings.temp_u[i] + 0.1  # (population, state, time)

up_u = np.zeros((pop + 1, N + 1, time_steps + 1))  # updated control

'''Leader: init: control: D; state: G'''
d = np.zeros((settings.G_n + 1, time_steps+1))
for i in range(0, settings.G_n+1):
    d[i, 0] = settings.temp_D[i] + 0.1

up_D = np.zeros((settings.G_n + 1, time_steps+1))
v0 = np.zeros((settings.G_n + 1, time_steps+1))  # leader's value function
leader_D = np.zeros((settings.G_n + 1, time_steps+1))  # leader's optimal determination

'''update gradient using FPK & HJB
def compute_grad(u, g):
    delta_new = FPK(u, g)
    v_new = HJB(u, delta_new, g)
    return delta_new, v_new
'''
'''update control'''
def u_new(u, g):
    for k in range(0, pop+1):
        for j in range(0, time_steps):
            for i in range(0, N + 1):
                #up_u[i, j] = (1/(delta[i, j] * settings.a1))*(delta[i, j] * g - 0.25 * (dx**2)*settings.beta*(v[int(settings.ipos[i]), j] - v[int(settings.ineg[i]), j])*(delta[int(settings.ipos[i]), j] - delta[int(settings.ineg[i]), j]))
                '''another update strategy'''
                #alpha = 0.2
                #up_u[i, j] = u[i, j] - alpha * 0.25 * (dx**2)*settings.beta*(v[int(settings.ipos[i]), j] - v[int(settings.ineg[i]), j])*(delta[int(settings.ipos[i]), j] - delta[int(settings.ineg[i]), j])
                '''another another update strategy'''
                up_u[k, i, j] = 1/(1+settings.omi)*u[k, i, j] - (settings.omi/(1+settings.omi))*(settings.a1*u[k, i, j] - g)
    return up_u



'''calculate error'''
def err(u, delta, g):
    '''Li(t)'''
    squared_err = 0
    for k in range(0, pop+1):
        for j in range(0, time_steps):
            for i in range(0, N + 1):
                squared_err += Li(u[k, i, j], delta[k, i, j], g, i) ** 2
                res = np.sqrt(squared_err)
    return res
'''1st time call FPK & HJB
delta1, v = compute_grad(u, g=0.25)
loss = err(u, delta1, g=0.25)
up_u = u_new(u, g=0.25)
loss_new = err(up_u, delta1, g=0.25)
'''

p_0 = np.zeros(10)
for x in range(0, 10, 1):
    p_0[x] = np.exp(-(x - 3) ** 2 / 2*1.1) / np.sqrt(2 * np.pi)
print('distribution', sum(p_0))

'''Gradient descent init'''
it = 1  # number of iteration
tol_L = settings.error

'''temp array for plot delta'''
temp_it_delta = np.zeros((settings.G_n+1, settings.I, pop+1, N+1, time_steps+1))
temp_it_value = np.zeros((settings.G_n+1, settings.I, pop+1, N+1, time_steps+1))

delta_new = np.zeros((pop+1, N+1, time_steps+1))
v_new = np.zeros((pop+1, N+1, time_steps+1))
it_err = np.zeros((settings.G_n+1, settings.I))
for q in range(0, settings.G_n+1):
    g = settings.G0 + q * settings.dG
    '''1st time call FPK & HJB'''
    #delta1, v = compute_grad(u, g)
    delta1 = FPK(u, g, it=1)
    v = HJB(u, delta1, g)
    loss = err(u, delta1, g)
    up_u = u_new(u, g)

    loss_new = err(up_u, delta1, g)
    print("loss and loss new", loss, loss_new)
    it_err[q, 0] = abs(loss_new-loss)
    it_err[q, 1] = abs(loss_new - loss)
    # while np.abs(loss_new - loss) > tol_L and it < I:
    while np.abs(loss_new - loss) > tol_L and it < settings.I:
        up_u = u_new(up_u, g)
        delta_new = FPK(up_u, g, it)
        v_new = HJB(up_u, delta_new, g)

        loss = loss_new
        loss_new = err(up_u, delta_new, g)
        temp_it_delta[q, it, :, :, :] = delta_new
        temp_it_value[q, it, :, :, :] = v_new
        it += 1
        print('Round %s, Reward %s, Diff RMSE %s'%(it, g, abs(loss_new - loss)))
        it_err[q, it] = abs(loss_new - loss)
    opt_control[q, :, :, :] = up_u
    opt_delta[q, :, :, :] = delta_new
    opt_value[q, :, :, :] = v_new

    np.save(f"{PATH}/delta.npy", opt_delta)
    np.save(f"{PATH}/temp_it_delta.npy", temp_it_delta)
    np.save(f"{PATH}/it_err.npy", it_err)


    '''reset value of the matrix'''
    delta1 = delta1 * 0
    v = v * 0
    up_u = up_u*0
    delta_new = delta_new*0
    v_new = v_new*0

    '''reset the while loop'''
    it = 1
    loss = 0

opt_u = np.zeros((settings.G_n+1, pop+1, N+1, time_steps+1))
opt_delta = np.load(PATH + f"/delta.npy")

for reward in range(0, settings.G_n+1):
    for pop in range(0, pop+1):
        for j in range(0, time_steps+1):
            for i in range(0, N + 1):
                #opt_u[reward, pop, i, j] = (- settings.beta*0.5/dx * (opt_value[reward, pop, int(settings.ipos[i]), j] - opt_value[reward, pop, int(settings.ineg[i]), j]))/settings.a1
                opt_u[reward, pop, i, j] = ((settings.G0 + reward * settings.dG)- (settings.beta * 0.5 / dx) * (opt_value[reward, pop, int(settings.ipos[i]), j] - opt_value[reward, pop, int(settings.ineg[i]), j]))/settings.a1

np.save(f"{PATH}/control.npy", opt_u)

it_opt_u = np.zeros((settings.G_n+1, settings.I, pop+1, N+1, time_steps+1))

for reward in range(0, settings.G_n+1):
    for it in range(0, settings.I):
        for pop in range(0, pop+1):
            for j in range(0, time_steps+1):
                for i in range(0, N + 1):
                    it_opt_u[reward, it, pop, i, j] = (- settings.beta*0.5/dx * (temp_it_value[reward, it, pop, int(settings.ipos[i]), j] - temp_it_value[reward, it, pop, int(settings.ineg[i]), j]))/settings.a1


np.save(f"{PATH}/it_control.npy", it_opt_u)

'''Leader's plots'''
# call function from Leader.py
D = Leader(PATH)
np.save(f"{PATH}/D.npy", D)








