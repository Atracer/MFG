'''partial differential equation: dLa/dvi'''
'''ddelta/dt = -(2*d*delta - r_i + beta u_i)* (ddelta/dr) - beta * delta * (du_i/dr) + (brown**2/2)* d2_delta '''

import math
from math import e
import numpy as np
import matplotlib.pyplot as plt
'''number of population'''
i = 3

'''we need to define 3 distributions!!!'''

'''normal probability density function'''
sigma1 = 0.31  # measure parameter: the shape of the curve
mu1 = 0.25  # mean: location parameter
x = np.linspace(-2, 2, 100)

delta = (1/(sigma1*((2*math.pi)**0.5))) * e**(-(1/(2*sigma1**2))*(x-mu1)**2)

d_delta = (1/(sigma1*((2*math.pi)**0.5))) * e**(-(1/(sigma1**2))*(x-mu1))
d2_delta = (1/(sigma1*((2*math.pi)**0.5))) * e**(-(1/(sigma1**2)))

print(d2_delta)
