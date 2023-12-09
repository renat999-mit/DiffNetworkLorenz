# Lorenz attractor nonlinear solution - training data generation

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

# xdot function
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot])

def dfdx(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    J = np.zeros((3,3))
    J[0][0] = -sigma # df1/dx
    J[0][1] = sigma # df1/dy
    J[0][2] = 0 # df1/dz
    J[1][0] = r-z # df2/dx
    J[1][1] = -1 # df2/dy
    J[1][2] = -x # df2/dz
    J[2][0] = y # df3/dx
    J[2][1] = x # df3/dy
    J[2][2] = -beta # df3/dz
    return J

if __name__ == "__main__":
    # nonlinear ODE - lorenz system
    # xdot = f(x)
    # x(0) = x0

    # params
    sigma = 10
    r = 28
    beta = 8/3

    porder = 2

    lorenz = np.load('dg_lorenz_dt100_p' + str(porder) + '.npz')

    xh = lorenz['xh'].T
    cs = lorenz['cs']
    t = lorenz['t']
    dt = t[1]-t[0]

    N = len(t)

    K = 1000 # number of training samples per element
    epsilon = 10e-3

    # loop through elements
    print('loop through elements')
    for n in range(0,N-1):
        x_tilde_n_current = np.zeros((K,3))
        x_tilde_np1_current = np.zeros((K,3))
        for k in range(0,K):
            x_tilde_n_k = epsilon*np.random.randn(3)
            x_n_k = xh[n] + x_tilde_n_k
            x_np1_k = x_n_k + f(x_n_k)*dt
            x_tilde_np1_k = x_np1_k - xh[n+1]
            x_tilde_n_current[k,:] = x_tilde_n_k
            x_tilde_np1_current[k,:] = x_tilde_np1_k
        
        if n == 0:
            
            x_tilde_n = x_tilde_n_current
            x_tilde_np1 = x_tilde_np1_current
            
        else:
            
            x_tilde_n = np.vstack((x_tilde_n,x_tilde_n_current))
            x_tilde_np1 = np.vstack((x_tilde_np1,x_tilde_np1_current))
            

    np.savez('diffusion-training-data', x_tilde_n=x_tilde_n, x_tilde_np1=x_tilde_np1)