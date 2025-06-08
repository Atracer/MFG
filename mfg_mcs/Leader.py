# LIBRARY
# vector manipulation
import numpy as np
# math functions
import math
import settings
# THIS IS FOR PLOTTING
# matplotlib inline
import matplotlib.pyplot as plt  # side-stepping mpl backend
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import warnings
#import Rethink_plot
'''Leader_Lax-Friedrichs Scheme'''
'''Input: mean field term: m(t = 0, r) and optimal control u from FPK'''
'''Output: optimal determination D(t)'''

warnings.filterwarnings("ignore")

def Leader(PATH):
    Gn = settings.G_n
    dg = settings.dG  # h
    g0 = settings.G0
    dt = 1 / settings.time_steps  # k
    time_steps = settings.time_steps

    D = np.zeros((Gn + 1, time_steps + 1))
    B = np.zeros((Gn + 1, time_steps + 1))
    C = np.zeros((Gn + 1, time_steps + 1))
    v0 = np.zeros((Gn + 1, time_steps + 1))

    opt_u = np.load(PATH + f"/control.npy")
    opt_delta = np.load(PATH + f"/delta.npy")
    agg_control = np.zeros((settings.G_n + 1, settings.pop + 1, settings.time_steps + 1))

    for reward in range(0, settings.G_n + 1):
        for pop in range(0, settings.pop + 1):
            for time in range(0, settings.time_steps + 1):
                for i in range(0, settings.n + 1):
                    agg_control[reward, pop, time] = opt_u[reward, pop, i, time] * opt_delta[reward, pop, i, time]  # expect state == 0
    agg_u = np.ones((settings.G_n + 1, settings.time_steps + 1))

    for reward in range(0, settings.G_n + 1):
        for time in range(0, settings.time_steps + 1):
            agg_u[reward, time] = agg_control[reward, 0, time] + agg_control[reward, 1, time]

    # print(agg_u[20:])
    np.save(f"{PATH}/U.npy", agg_u)

    agg_reputation = np.zeros((settings.G_n + 1, settings.time_steps + 1))
    for reward in range(0, settings.G_n + 1):
        for time in range(0, settings.time_steps + 1):
            for i in range(0, settings.n + 1):
                agg_reputation[reward, time] = opt_delta[reward, 0, i, time] * i + opt_delta[reward, 1, i, time] * i
    #print(agg_reputation[20:])
    np.save(f"{PATH}/Rep.npy", agg_reputation)

    '''Initial Condition'''
    for i in range(0, Gn+1):
        v0[i, time_steps] = settings.v0[i]

    '''call FPK to obtain control and mean field'''
    lamba = dt / dg
    ran = (dt)/(dg**2)

    # parameter B, C
    for j in range(0, time_steps):
        for i in range(0, Gn+1):
            B[i, j] = (settings.omega1**2 * (g0 + dg *i)**2) / (4 * settings.tau * agg_reputation[i, j])
            C[i, j] = settings.omega2 * agg_reputation[i,j] - settings.omega1*(g0 + i * dg) * agg_u[i,j]
    print(B[1,1], C[1,1])
    for j in range(time_steps, 0, -1):
        for i in range(0, Gn+1):
            v0[i, j - 1] = 0.5 * (v0[int(settings.lipos[i]), j] + v0[int(settings.lineg[i]), j]) \
                           - lamba * C[i, j] * (v0[int(settings.lipos[i]), j] - v0[int(settings.lineg[i]), j])\
                           + ran * B[i, j] * (v0[int(settings.lipos[i]), j] - v0[int(settings.lineg[i]), j]) ** 2
    print(v0[3,3])
    for j in range(0, time_steps):
        for i in range(0, Gn+1):
            D[i, j] = - (settings.omega1 * (g0 + i * dg))/(2 * settings.tau * agg_reputation[i,j]) * (v0[int(settings.lipos[i]), j] - v0[int(settings.lineg[i]), j])/dg

    return D
