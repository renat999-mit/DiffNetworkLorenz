# train SDE diffusion
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

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

def naive_model(alpha, lambda_max = 0.95):
    mu = alpha * np.sqrt(2*lambda_max)
    return mu*xtilde[n,:]


porder = 2

lorenz = np.load('dg_lorenz_dt100_p' + str(porder) + '.npz')

xh = lorenz['xh'].T
x_n = xh[0:-1,:]
cs = lorenz['cs']
t = lorenz['t']
dt = t[1]-t[0]

N = len(t)
epsilon = 10e-3
sigma = 10
r = 28
beta = 8/3

# initial conditions
x_EM = np.zeros((N,3))
xtilde = np.zeros((N,3))
x_alpha = np.zeros((N,3))
x_EM[0,:] = [-8.67139571762,4.98065219709,25]
x_alpha[0,:] = [-8.67139571762,4.98065219709,25]

xtilde[0,:] = epsilon*np.random.randn(3)

plt.figure(4,figsize=(6,4))

alphas = np.array([0.5,1.0,1.5])

for alpha in alphas:
    # time integration
    for n in range(N-1):
        tn = t[n]
        dW = np.sqrt(dt) * np.random.randn(N)

        # Euler-Maruyama method
        x_EM[n+1,:] = x_EM[n,:] + f(x_EM[n,:])*dt

        # alpha method
        xtilde[n+1,:] = xtilde[n,:] + np.matmul(dfdx(x_alpha[n,:]),xtilde[n,:])*dt + naive_model(alpha)*dW[n]
        x_alpha[n+1,:] = xh[n+1,:] + xtilde[n+1,:]

    plt.plot(t,((np.abs(xtilde[:,0]))),label=r"$\alpha = $" + str(alpha))
    plt.xlabel("t")
    plt.yscale("log")
    plt.ylabel(r"$|\tilde{\mathbf{x}}_\alpha|$")
    #plt.title(r"$\alpha$ = " + str(alpha))
    plt.legend(loc='upper left')
    #plt.ylim(-50,50)
    plt.xlim(-5,5)
    plt.grid()

# plt.subplot(1,3,2)
# plt.plot(t,xtilde[:,1],label=r"$\tilde{y}_\alpha$")
# plt.xlabel("t")
# plt.ylabel(r"$\tilde{\mathbf{y}}_\alpha$")
# #plt.legend()
# plt.ylim(-50,50)
# plt.grid()

# plt.subplot(1,3,3)
# plt.plot(t,xtilde[:,2],label=r"$\tilde{z}_\alpha$")
# plt.xlabel("t")
# plt.ylabel(r"$\tilde{\mathbf{z}}_\alpha$")
# #plt.legend()
# plt.ylim(-50,50)
# plt.grid()
# plt.tight_layout()

plt.savefig('alpha-x.png', format='png', dpi=300,transparent=True,bbox_inches='tight')
plt.show()

