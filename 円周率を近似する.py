

import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math

import scipy.linalg as linalg
from scipy.optimize import minimize_scalar

from scipy.optimize import newton
N = 100000
x = np.random.uniform(0.0,1.0,N)
y = np.random.uniform(0.0,1.0,N)

ins = []
out = []
count = 0
#ins=set(ins)
#out=set(out)
for i in range(N):
    if math.hypot(x[i],y[i])<1:
        ins.append((x[i],y[i]))
        count += 1
    else:
        out.append((x[i],y[i]))
plt.figure(figsize=(8,7))
ins_x,ins_y = zip(*ins)
out_x,out_y = zip(*out)
plt.plot(ins_x,ins_y,'o',label='Inside')
plt.plot(out_x,out_y,'o',label='Outside')
plt.legend()
#plt.grid(True)
plt.show()
print('円周率の近似',4.0 * count / N)