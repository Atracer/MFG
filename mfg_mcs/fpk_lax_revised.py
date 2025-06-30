# LIBRARY
# matplotlib inline
import matplotlib.pyplot as plt  # side-stepping mpl backend
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
# vector manipulation
import numpy as np
# math functions
import math
import settings
from settings import fi, Li
from scipy.stats import poisson

import warnings

'''FPK_Lax-Friedrichs Scheme'''
'''Input: mean field term: m(t = 0, r)'''
'''Input: control u(t = 0, r)'''
'''Output: mean field term: M(t, r)'''

warnings.filterwarnings("ignore")

def FPK(u, g, it):
    N = settings.n
    dx = settings.R/settings.n  # h
    time_steps = settings.time_steps
    dt = 1/time_steps
    pop = settings.pop+1

    f = np.zeros((2, N + 1, time_steps + 1))
    '''init mean field term'''
    delta = np.zeros((2, N + 1, time_steps + 1))
    '''plot 0624'''

    r = np.linspace(0, settings.R, N+1)  # all the state

    '''set init mean field term'''
    x = np.arange(0, settings.n + 1, 1)

    # poisson distribution data for y-axis
    mean1 = 10
    mean2 = 20
    y1 = poisson.pmf(x, mean1)
    y2 = poisson.pmf(x, mean2)

    for i in range(0, N + 1):
        '''delta[pop, state, time]'''
        delta[0, i, 0] = y1[i]
        delta[1, i, 0] = y2[i]

    #归一化
    delta[0, :, 0] /= np.sum(delta[0, :, 0]) + 1e-8
    delta[1, :, 0] /= np.sum(delta[1, :, 0]) + 1e-8


    #print('sum delta', sum(delta[0, :, 0]), sum(delta[1, :, 0]))
    #print('init delta', delta[0, :, 0], delta[1, :, 0])

        # delta[i, time_steps] = settings.init_m1[i]

    '''iteration'''
    lamba = dt / dx
    ran = dt/(8*dx**2)
    #print(delta)

    for k in range(0, pop):
        for j in range(0, time_steps + 1):
            for i in range(0, N + 1):
                # 裁剪，避免震荡放大
                f_raw = fi(u[k, i, j], delta[k, i, j], g, i, k)
                f[k, i, j] = np.clip(f_raw, -1.0, 1.0)


    '''plot 0624'''
    #plt.figure(figsize=(12, 9))

    for k in range(0, pop):
        for j in range(0, time_steps):
            '''init at each time step'''

            for i in range(0, N + 1):
                if u[k, i, j] == 0:
                    delta[k, i, j + 1] = delta[k, i, j]
                else:
                    delta[k, i, j + 1] = 0.5*(delta[k, int(settings.ipos[i]), j] + delta[k, int(settings.ineg[i]), j]) \
                            - (f[k, int(settings.ipos[i]), j]*delta[k, int(settings.ipos[i]), j] - f[k, int(settings.ineg[i]), j]*delta[k, int(settings.ineg[i]), j])*lamba / 2
                     # 保证非负
                    delta[k, i, j + 1] = max(delta[k, i, j + 1], 0.0)
                    
            # 每个时间步归一化
            delta_sum = np.sum(delta[k, :, j + 1])
            if delta_sum > 0:
                delta[k, :, j + 1] /= delta_sum
            
    return delta


'''plot FPK: FPK_g_0.png
#plot 0624
            if j % 20 == 0:  # and it % 3
                plt.plot(r, delta[1, :, j], linestyle='--', label='Crowd 2, t = %s'%j)
                plt.plot(r, delta[0, :, j],  marker='o', label='Crowd 1, t = %s'%j)
        plt.xlabel("state")
        plt.legend(loc='upper left')
        plt.title("FPK g=" + str(g))
        plt.savefig('FPK_g_%s.png'%g, dpi=300)
        plt.show()
u = np.ones((settings.pop + 1, settings.n + 1, settings.time_steps + 1)) * 0.2
delta = FPK(u, g=0.2, it=1)
'''

'''plot FPK: 3D
# 0621 plot control with reward in term of pop
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')


state = np.linspace(0, 1, settings.n+1)
print('state', settings.n)
time_plot = np.linspace(0, 1, settings.time_steps+1)

X, Y = np.meshgrid(state, time_plot)
pop1 = delta[0, :, :].transpose()
pop2 = delta[1, :, :].transpose()
surf1 = ax.plot_surface(X, Y, pop1, cmap = plt.cm.cividis, label='Crowd 1')
surf2 = ax.plot_surface(X, Y, pop2,  label='Crowd 2')
surf1._facecolors2d = surf1._facecolor3d
surf1._edgecolors2d = surf1._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d
surf2._edgecolors2d = surf2._edgecolor3d
# Set axes label
ax.set_xlabel('State', labelpad=20, fontsize=20)
ax.set_ylabel('Time', labelpad=20, fontsize=20)
ax.set_zlabel('State distribution', labelpad=20, fontsize=20)
#fig.colorbar(surf, shrink=0.5, aspect=8)
ax.legend(loc='upper left', prop={'size': 20})

plt.savefig('FPK_g_02_3d.png', dpi=300)
plt.show()
'''


