# THIS IS FOR PLOTTING
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
import settings
from Leader import Leader1
import Rethink_plot

'''0702 leader's game: failed??'''
def leader_0702(u, delta):
    Gn = settings.G_n
    dg = settings.dG  # h
    dt = 1 / settings.time_steps  # k
    time_steps = settings.time_steps

    lamda = np.zeros((settings.G_n+1, settings.time_steps+1))
    #reward = np.zeros((settings.G_n+1, settings.time_steps+1))  # state: optimal reward
    reward = np.zeros(settings.time_steps + 1)  # state: optimal reward
    deter = np.zeros((settings.G_n+1, settings.time_steps+1))  # control

    '''init control D'''
    for i in range(0, Gn+1):
        deter[i, time_steps] = settings.D[i]
    '''init lambda'''
    for i in range(0, Gn+1):
        lamda[i, time_steps] = settings.v0[i]

    for j in range(0, time_steps):
        for i in range(0, Gn + 1):
            lamda[i, j+1] = lamda[i, j] + dt*u[i,j] + 0.5*dt*lamda[i, j]*settings.omega_1/settings.tau\
                            *(settings.eta + lamda[i, j]*settings.omega_1*(i*dg)) + lamda[i, j]*settings.omega_1*u[i,j]
                #0.5 * (lamda[int(settings.lipos[i]), j] + lamda[int(settings.lineg[i]), j]) \
                #           + dt * (u[i,j] + lamda[i,j] * settings.omega_1 * deter[i,j])

    for j in range(0, time_steps+1):
        for i in range(0, Gn + 1):
            deter[i, j] = (-lamda[i, j] * settings.omega_1 * (i*dg) - settings.eta)/(2*settings.tau)
    for j in range(0, time_steps):
        for i in range(0, Gn + 1):
            reward[j+1] = reward[j] + dt*settings.omega_1*reward[j]*(deter[i,j]-u[i,j])+ settings.omega_2*delta[i,j]
                #0.5 * (reward[int(settings.lipos[i]), j] + reward[int(settings.lineg[i]), j])\
                #               + dt * (i*dg) + dt*settings.w1*(deter[i,j] - u[i,j])*(i*dg) \
                #               + dt * settings.omega_2 * delta[i,j]
    return deter, reward
'''0621 here another way '''
def Leader1(D, U, Delta):
    Gn = settings.G_n
    dg = (settings.Gm + 0.1 - settings.G0) / settings.G_n # h
    dt = 1/settings.time_steps  # k
    time_steps = settings.time_steps
    #print('time step in leader', time_steps)
    A = np.zeros((Gn+1, time_steps + 1))
    B = np.zeros((Gn+1, time_steps + 1))
    v0 = np.zeros((Gn+1, time_steps + 1))

    #print('v0 in leader', len(v0))

    '''Initial Condition'''
    temp_v = np.ones(Gn+1)
    for i in range(0, Gn+1):
        v0[i, time_steps] = temp_v[i]
    #print('v0 in leader', len(v0), len(v0[0]))
    #print('v0 in leader', v0)

    '''call FPK to obtain control and mean field'''
    lamba = dt / dg
    ran = (dt)/(4*dg**2)

    # parameter A, B, C
    for j in range(0, time_steps):
        for i in range(0, Gn+1):
            A[i, j] = D[i, j] * i * dg + settings.eta * D[i, j] + settings.eta * (D[i, j] ** 2)
            B[i, j] = settings.omega_1 * (D[i, j] - U[i,j]) * i * dg + settings.omega_2 * Delta[i, j]
    #print('AB in leader', len(A), len(A[0]), len(B), len(B[0]))

    for j in range(time_steps, 0, -1):
        for i in range(0, Gn+1):
            v0[i, j - 1] = 0.5 * (v0[int(settings.lipos[i]), j] + v0[int(settings.lineg[i]), j]) \
                        + 0.5 * lamba * B[i, j] * (v0[int(settings.lipos[i]), j] - v0[int(settings.lineg[i]), j]) + dt * A[i,j]
                           #+ ran * (v0[int(settings.liipos[i]), j] - 2 * v0[int(settings.lorg[i]), j] + v0[int(settings.liineg[i]), j])
            #print('value leader:', v0)

    return v0

'''init'''
time_steps = settings.time_steps
#N = settings.n
#dx = settings.R/N
#pop = 1
dg = (settings.Gm + 0.1 - settings.G0) / settings.G_n

'''Leader: init: control: D; state: G'''
d = np.zeros((settings.G_n + 1, time_steps+1))
for i in range(0, settings.G_n+1):
    d[i, 0] = settings.temp_D[i] + 0.1
#print('d in leader_deter', len(d), len(d[0]))

up_D = np.zeros((settings.G_n + 1, time_steps+1))
v0 = np.zeros((settings.G_n + 1, time_steps+1))  # leader's value function
leader_D = np.zeros((settings.G_n + 1, time_steps+1))  # leader's optimal determination

'''update gradient using Leader'''
agg_u = np.ones((settings.G_n+1, time_steps+1))
agg_delta = np.ones((settings.G_n+1, time_steps+1))


for reward in range(0, settings.G_n+1):
    for time in range(0, time_steps+1):
        agg_u[reward, time] = Rethink_plot.U[reward, 0, time] + Rethink_plot.U[reward, 1, time]
        agg_delta[reward, time] = Rethink_plot.Delta1[reward, 0, time] + Rethink_plot.Delta1[reward, 1, time]

'''update control D'''

def D_new(D):
    for j in range(0, time_steps+1):
        for i in range(0, settings.G_n + 1):
            up_D[i, j] = 1/(1+settings.omi)*D[i, j] - (settings.omi/(1+settings.omi))*(settings.eta + 2 * settings.tau * D[i, j])
    return up_D


'''calculate error'''
def err(U, D):
    '''L0(g,t)'''
    squared_err = 0
    for j in range(0, time_steps):
        for i in range(0, settings.G_n + 1):
            '''Leader's L'''
            squared_err += (U[i, j] * i * dg + settings.eta * D[i, j] + settings.tau * D[i, j]**2)**2
            # res = np.sqrt(np.mean(squared_err))
            res = np.sqrt(squared_err)
    return res
'''1st time call FPK & HJB'''
'''first iteration'''
#print('leader1 v0', len(v0))
v0 = Leader1(d, agg_u, agg_delta)
#print('control', U)

loss = err(agg_u, d)
up_D = D_new(d)
#print('first deter update', up_D)
loss_new = err(agg_u, up_D)


'''Gradient descent init'''
it = 1  # number of iteration
I = 100  # total iteration
tol_L = 0.1

while np.abs(loss_new - loss) > tol_L and it < I:
    up_D = D_new(up_D)
    #print('update u', up_u)
    v_new = Leader1(up_D, agg_u, agg_delta)
    #print('new value', v_new)
    loss = loss_new
    loss_new = err(agg_u, up_D)
    #print('loss new', loss_new)
    it += 1
    print('Round %s Diff RMSE %s'%(it, abs(loss_new - loss)))

#print('leader inter', up_D)

'''
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

state = np.arange(0, 11, 1)
time_plot = np.arange(0, 21, 1)

X, Y = np.meshgrid(state, time_plot)
Z = up_D.transpose()

surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

# Set axes label
ax.set_xlabel('State', labelpad=20)
ax.set_ylabel('Time', labelpad=20)
ax.set_zlabel('State distribution', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()
'''