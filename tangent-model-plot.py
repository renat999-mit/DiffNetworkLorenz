# plot TLM

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.special import jacobi

def plegendre(x,porder):
    
    try:
        y = np.zeros((len(x),porder+1))
        dy = np.zeros((len(x),porder+1))
        ddy = np.zeros((len(x),porder+1))
    except TypeError: # if passing in single x-point
        y = np.zeros((1,porder+1))
        dy = np.zeros((1,porder+1))
        ddy = np.zeros((1,porder+1))

    y[:,0] = 1
    dy[:,0] = 0
    ddy[:,0] = 0

    if porder >= 1:
        y[:,1] = x
        dy[:,1] = 1
        ddy[:,1] = 0
    
    for i in np.arange(1,porder):
        y[:,i+1] = ((2*i+1)*x*y[:,i]-i*y[:,i-1])/(i+1)
        dy[:,i+1] = ((2*i+1)*x*dy[:,i]+(2*i+1)*y[:,i]-i*dy[:,i-1])/(i+1)
        ddy[:,i+1] = ((2*i+1)*x*ddy[:,i]+2*(2*i+1)*dy[:,i]-i*ddy[:,i-1])/(i+1)

    # return y,dy,ddy
    return y,dy

tilde = False

dg_lorenz = np.load('dg_lorenz_dt100_p2.npz')
xh = dg_lorenz['xh']
xh_tilde = dg_lorenz['xh_tilde']
t = dg_lorenz['t']

# dg_lorenz_filtered = np.load('dg_lorenz_dt100_p2_stoch.npz')
# xhbar = dg_lorenz_filtered['xhbar']
# xhbar_tilde = dg_lorenz_filtered['xhbar_tilde']

fig1 = plt.figure(figsize=(10,2))
ax2 = fig1.add_subplot(1,3,1)
ax2y = fig1.add_subplot(1,3,2)
ax2z = fig1.add_subplot(1,3,3)

ax2.plot(t,(xh_tilde[0,:]),color='C0')
ax2y.plot(t,(xh_tilde[1,:]),color='C1')
ax2z.plot(t,(xh_tilde[2,:]),color='C2')
ax2.set_ylabel(r"$(\tilde{\mathbf{x}})$")
ax2.set_ylim(-300,300)
ax2y.set_ylim(-300,300)
ax2z.set_ylim(-300,300)
plt.show()

fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1.plot(*xh_tilde,label=r"$\tilde{\mathbf{x}}$")
ax1.set_xlabel(r"$\tilde{x}$")
ax1.set_ylabel(r"$\tilde{y}$")
ax1.set_zlabel(r"$\tilde{z}$")
ax1.set_xlim(-300,300)
ax1.set_ylim(-300,300)
ax1.set_zlim(-300,300)
#ax1.legend()

plt.savefig('unstable-3d.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
plt.show()


# N = len(t)
# dt = t[1] - t[0]
# porder = 0

# xi, w = scipy.special.roots_legendre(porder+1)
# phi, dphi = plegendre(xi,porder)

# fig1 = plt.figure(figsize=(10,2))

# ax2 = fig1.add_subplot(1,3,1)
# ax2y = fig1.add_subplot(1,3,2)
# ax2z = fig1.add_subplot(1,3,3)

# # ax3 = fig1.add_subplot(3,2,2)
# # ax3y = fig1.add_subplot(3,2,4)
# # ax3z = fig1.add_subplot(3,2,6)

# ax2.plot(t,np.log(xh_tilde[0,:]),color='C0')
# ax2y.plot(t,np.log(xh_tilde[1,:]),color='C1')
# ax2z.plot(t,np.log(xh_tilde[2,:]),color='C2')
# ax2.set_ylabel(r"$\log(\tilde{\mathbf{x}})$")
# ax2.set_xlabel(r"$t$")
# ax2y.set_xlabel(r"$t$")
# ax2z.set_xlabel(r"$t$")
# # ax2.set_yscale("log")
# # ax2y.set_yscale("log")
# # ax2z.set_yscale("log")
# # ax2.set_ylim(-300,300)
# # ax2y.set_ylim(-300,300)
# # ax2z.set_ylim(-300,300)

# # ax3.plot(t,xhbar_tilde[0,:],color='C0')
# # ax3y.plot(t,xhbar_tilde[1,:],color='C1')
# # ax3z.plot(t,xhbar_tilde[2,:],color='C2')
# # ax3.set_title(r"$\tilde{\overline{\mathbf{x}}}$")
# # ax3z.set_xlabel(r"$t$")
# # ax3.set_ylim(-300,300)
# # ax3y.set_ylim(-300,300)
# # ax3z.set_ylim(-300,300)


# #plt.savefig('plots/dg-lorenz-tangent-linear.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
# plt.show()

# if 0:
#     # Lyapunov covariant vectors

#     lambda_ = 0.95

#     dg_lorenz_perturbed = np.load('dg_lorenz_dt100_p2_perturbed.npz')
#     xh_perturbed = dg_lorenz_perturbed['xh']
#     cs_perturbed = dg_lorenz_perturbed['cs']

#     fig0 = plt.figure(figsize=(12,4))
#     ax1 = fig0.add_subplot(1,2,1)
#     ax1.semilogy(t,np.sqrt(np.abs(xh_perturbed[0,:]-xh[0,:])**2),'C0')
#     ax1.semilogy(t,np.sqrt(np.abs(xh_perturbed[1,:]-xh[1,:])**2),'C1')
#     ax1.semilogy(t,np.sqrt(np.abs(xh_perturbed[2,:]-xh[2,:])**2),'C2')
#     ax1.semilogy(t,10**-1.5*np.exp(lambda_*t),'k--',label=r"$\exp(\lambda t)$ with $\lambda$ = " + str(lambda_))
#     ax1.legend(loc='lower right')
#     ax1.set_xlabel(r"$t$")
#     ax1.set_ylabel(r"$||\mathbf{x}_h^p - \mathbf{x}_h||$")
#     ax1.set_ylim(10**-6,10**2)

#     phi = xh_tilde*np.exp(-lambda_*t)

#     ax2 = fig0.add_subplot(1,2,2)

#     ax2.plot(t,phi[0,:],label=r"$x$")
#     ax2.plot(t,phi[1,:],label=r"$y$")
#     ax2.plot(t,phi[2,:],label=r"$z$")
#     ax2.set_ylabel(r"$\mathbf{\phi}$")
#     ax2.set_xlabel(r"$t$")
#     ax2.legend()
#     ax2.set_ylim(-50,50)

#     #plt.savefig('plots/dg-lorenz-tangent-linear-lyapunov.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
#     plt.show()





