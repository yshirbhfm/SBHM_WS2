# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:08:56 2022

@author: Lab KYLO
"""

import numpy as np
import scipy.integrate as integrate # scipy提供的數值積分工具
import matplotlib.pyplot as plt     # for visualization
import math

# Parameter for absorption
d = [0,5,5,80,600000] # nm
absor_2w = [1] * 5
N2 = [1, 1.4673+0.0020319j, 4.219164564+0.03673713j,4.200669376+0.07498804876j,4.219164564+0.03673713j]
    
   
a = math.exp(10)
        
        
f1 = lambda x : np.exp(-2*np.pi*N2[4].imag*x/522)

xi = np.linspace(0,50,10000)  # interval of integration

print(xi[0],f1(xi)[0])   # starting point of the integration (fist element of xi and f1)
print(xi[-1],f1(xi)[-1]) # ending point of the integration (last element of xi and f1)


# Let's visualize the equation and its integration first
plt.plot(xi,f1(xi),'b-')
# plt.vlines(xi[0],-10,f1(xi)[0],colors='r')
# plt.vlines(xi[-1],-10,f1(xi)[-1],colors='r')
# plt.hlines(0,xi[0],xi[-1],colors='r')
# plt.xlim(-0.1,10.1)
# plt.ylim(0,102)
plt.show()


area = integrate.quad(f1, 0, 50)
print(area)