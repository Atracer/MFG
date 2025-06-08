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
from settings import Li, fi
# THIS IS FOR PLOTTING
# matplotlib inline
import matplotlib.pyplot as plt  # side-stepping mpl backend
import warnings
from FPK_Lax import FPK

'''HJB_Lax-Friedrichs Scheme'''
'''Input: value function V(t = 0, r)'''
'''Input: control u(t, r) from FPK_Lax-F'''
'''Input: mean field m(t = 0, r) from FPK_Lax_F'''

'''Output: mean field term: M(t, r)'''

warnings.filterwarnings("ignore")

def HJB(u, delta, g):
    N = settings.n  # discrete numbers of state
    dx = settings.R/N  # h
    dt = 1/settings.time_steps  # k
    time_steps = settings.time_steps
    pop = settings.pop+1
    '''plot 0624'''
    #x = np.linspace(0, settings.R, N + 1)

    v = np.zeros((pop, N + 1, time_steps + 1))
    L = np.zeros((pop, N + 1, time_steps + 1))
    f = np.zeros((pop, N + 1, time_steps + 1))
    dLdm = np.zeros((pop, N + 1, time_steps + 1))

    '''Rethink: value function init V(x, T)'''
    for k in range(0, pop):
        for i in range(0, N + 1):
            v[k, i, time_steps] = settings.v_hjb[i]

    '''FPK_Lax: Call control'''
    #delta = FPK(u, g)

    '''iteration'''
    lamba = dt / dx
    # ran = (settings.brown[0]**2*dt)/(8*dx**2)
    '''plot 0624'''
    #plt.figure(figsize=(12, 9))
    for k in range(0, pop):
        for j in range(time_steps, 0, -1):
            for i in range(0, N + 1):
                L[k, i, j] = Li(u[k, i, j], delta[k, i, j], g, i)

                f[k, i, j] = fi(u[k, i, j], delta[k, i, j], g, i, k)
                dLdm[k, i, j] = settings.a2

                v[k, i, j - 1] = 0.5*(v[k, int(settings.ipos[i]), j] + v[k, int(settings.ineg[i]), j])\
                            - 0.5 * lamba * (f[k, i, j] + settings.beta1 * delta[k, i, j]) * (v[k, int(settings.ipos[i]), j] - v[k, int(settings.ineg[i]), j])\
                            + dt * (L[k, i, j] + delta[k, i, j] * dLdm[k, i, j])
            '''plot 0624'''
            #if j % 5 == 0:  # and it % 3
                #plt.plot(x, v[1, :, j], linestyle='--', label='t = %s'%j)
                #plt.plot(x, delta[0, :, j],  marker='o', label='t = %s'%j)
        #plt.xlabel("state")
        #plt.legend(loc='upper left')
        #plt.title("HJB g=" + str(g))
        #plt.show()
    return v

'''Test HJB
u = np.ones((settings.pop + 1, settings.n + 1, settings.time_steps + 1)) * 0.9
delta = np.zeros((settings.pop + 1, settings.n + 1, settings.time_steps + 1))
for k in range(0, settings.pop+1, 1):
    for j in range(0, settings.time_steps+1, 1):
        for i in range(0, settings.n, 1):
            delta[k, i, j] = np.exp(-(i - 3) ** 2 / 2*1.1) / np.sqrt(2 * np.pi)

v = HJB(u, delta, g=0.25)
'''