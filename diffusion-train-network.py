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

import tensorflow as tf
from Network_DietrichBased import (
                                    SDEIdentification,
                                    ModelBuilder,
                                    SDEApproximationNetwork
                                  )
 
################################ Load data ####################################

training_data = np.load('diffusion-training-data.npz')
x_tilde_n = training_data['x_tilde_n']
x_tilde_np1 = training_data['x_tilde_np1']

porder = 2

lorenz = np.load('dg_lorenz_dt100_p' + str(porder) + '.npz')

xh = lorenz['xh'].T
x_n = xh[0:-1,:]
cs = lorenz['cs']
t = lorenz['t']
dt = t[1]-t[0]

# SPECIFY IF YOU WANT TO TRAIN AND SAVE THE RESULTS
train = False
save = False

######################### Network parameters ##################################

n_layers = 3 #Number of hidden layers
n_dim_per_layer = 10 #Neurons per layer

n_dimensions = 3 #Spatial dimension 

ACTIVATIONS = tf.nn.relu #Activation function
VALIDATION_SPLIT = .2 # 80% for training, 20% for testing
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
N_EPOCHS = 50

# use diagonal sigma matrix
diffusivity_type = "diagonal"

# define the neural network model we will use for identification
encoder = ModelBuilder.diff_network(
                                    n_input_dimensions=int(2*n_dimensions), #Takes xn and tilde_xn
                                    n_output_dimensions=n_dimensions,
                                    n_layers=n_layers,
                                    n_dim_per_layer=n_dim_per_layer,
                                    name="diff_net",
                                    activation=ACTIVATIONS,
                                    diffusivity_type=diffusivity_type)
#encoder.summary()

#dictionary with jacobian parameters
jac_par = {'sigma': 10, 'r': 28, 'beta': 8/3}

if train:

    # Prepare the input tensor
    for n_point, nominal_point in enumerate(x_n):
        
        xn_matrix = np.tile(nominal_point,(1000,1))
        
        slice_start = int(n_point*1000)
        slice_end = int(slice_start + 1000)
        
        x_tilde_n_slice = x_tilde_n[slice_start:slice_end,:]
        x_tilde_n1_slice = x_tilde_np1[slice_start:slice_end,:]
        
        new_block = np.hstack((xn_matrix,x_tilde_n_slice,x_tilde_n1_slice))
        
        if n_point == 0:
            
            full_data = new_block
            
        else:
            
            full_data = np.vstack((full_data,new_block))
            
    n_pts = x_tilde_n.shape[0]

    step_sizes = np.zeros(n_pts) + dt

    model = SDEApproximationNetwork(sde_model=encoder,
                                    step_size=dt,
                                    jac_par=jac_par,
                                    method="euler",
                                    diffusivity_type=diffusivity_type)

    model.compile(optimizer=tf.keras.optimizers.Adamax())

    sde_i = SDEIdentification(model=model)

    xn = full_data[:,0:3]
    tilde_xn = full_data[:,3:6]
    tilde_xn1 = full_data[:,6:9]

    hist = sde_i.train_model(xn, tilde_xn, tilde_xn1, step_size=step_sizes,
                            validation_split=VALIDATION_SPLIT,
                            n_epochs=N_EPOCHS,
                            batch_size=BATCH_SIZE)

    plt.figure(16,figsize=(6,4))
    plt.title(r"Diagonal $\Sigma$")
    plt.plot(hist.history["loss"], label='Training')
    plt.plot(hist.history["val_loss"], label='Validation')
    plt.ylim([np.min(hist.history["loss"])*1.1, np.max(hist.history["loss"])])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training.png')
    plt.show()

    # plt.figure(17,figsize=(6,4))
    # plt.title(r"Diagonal $\Sigma$")
    # plt.plot(hist.history["accuracy"], label='Training')
    # plt.plot(hist.history["val_accuracy"], label='Validation')
    # plt.ylim([np.min(hist.history["accuracy"])*1.1, np.max(hist.history["accuracy"])])
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.savefig('training-accuracy.png')
    # plt.show()

file_path = 'Trained_Dietrich'
file_path += '/' + diffusivity_type + '/'
file_path += f'HL{n_layers}_'
file_path += f'N{n_dim_per_layer}_'
file_path += 'LReLU_'
file_path += 'LR1e-1_'
file_path += f'BS{BATCH_SIZE}_'
file_path += f'EP{N_EPOCHS}/'

if save:
    model.save_weights(file_path, overwrite=True, save_format=None, options=None)

################## TEST

model = SDEApproximationNetwork(sde_model=encoder,
                                    step_size=dt,
                                    jac_par=jac_par,
                                    method="euler",
                                    diffusivity_type=diffusivity_type)

model.load_weights(file_path)

sde_i = SDEIdentification(model=model)

N = len(t)
epsilon = 10e-3
sigma = 10
r = 28
beta = 8/3

# initial conditions
x_EM = np.zeros((N,3))
xtilde = np.zeros((N,3))
xtilde_NN = np.zeros((N,3))
x_NN = np.zeros((N,3))
x_alpha = np.zeros((N,3))
x_EM[0,:] = [-8.67139571762,4.98065219709,25]
x_NN[0,:] = [-8.67139571762,4.98065219709,25]
x_alpha[0,:] = [-8.67139571762,4.98065219709,25]

xtilde[0,:] = epsilon*np.random.randn(3)
xtilde_NN[0,:] = xtilde[0,:]

# time integration
for n in range(N-1):
    tn = t[n]
    dW = np.sqrt(dt) * np.random.randn(N)

    # Euler-Maruyama method
    x_EM[n+1,:] = x_EM[n,:] + f(x_EM[n,:])*dt

    # alpha method
    alpha = 1.0
    xtilde[n+1,:] = xtilde[n,:] + np.matmul(dfdx(x_alpha[n,:]),xtilde[n,:])*dt + naive_model(alpha)*dW[n]
    x_alpha[n+1,:] = xh[n+1,:] + xtilde[n+1,:]

    # NN method
    xtilde_NN[n+1,:] = sde_i.sample_tilde_xn1(x_NN[n,:], xtilde_NN[n,:], dt, jac_par, diffusivity_type)
    x_NN[n+1,:] = xh[n+1,:] + xtilde_NN[n+1,:]


# plt.figure(3,figsize=(12,4))
# plt.subplot(1,3,1)
# plt.plot(t,np.log(np.abs(xtilde[:,0])),label=r"$\tilde{x}_\alpha$")
# plt.plot(t,np.log(np.abs(xtilde_NN[:,0])),label=r"$\tilde{x}_{NN}$")
# plt.xlabel("t")
# plt.yscale("log")
# #plt.title(r"$\alpha$ = " + str(alpha))
# plt.ylabel(r"$\log(|\tilde{\mathbf{x}}|)$")
# plt.legend()
# plt.grid()

# plt.subplot(1,3,2)
# plt.plot(t,np.log(np.abs(xtilde[:,1])),label=r"$\tilde{y}_\alpha$")
# plt.plot(t,np.log(np.abs(xtilde_NN[:,1])),label=r"$\tilde{y}_{NN}$")
# plt.xlabel("t")
# plt.yscale("log")
# plt.ylabel(r"$\log(|\tilde{\mathbf{y}}|)$")
# plt.legend()
# plt.grid()

# plt.subplot(1,3,3)
# plt.plot(t,np.log(np.abs(xtilde[:,2])),label=r"$\tilde{z}_\alpha$")
# plt.plot(t,np.log(np.abs(xtilde_NN[:,2])),label=r"$\tilde{z}_{NN}$")
# plt.xlabel("t")
# plt.yscale("log")
# plt.ylabel(r"$\log(|\tilde{\mathbf{z}}|)$")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig('logtildes.png', format='png', dpi=300)
# plt.show()

plt.figure(4,figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(t,(xtilde[:,0]),label=r"$\tilde{x}_\alpha$")
plt.plot(t,(xtilde_NN[:,0]),label=r"$\tilde{x}_{NN}$")
plt.xlabel("t")
#plt.title(r"$\alpha$ = " + str(alpha))
plt.ylabel(r"$\tilde{\mathbf{x}}$")
plt.legend()
plt.grid()

plt.subplot(1,3,2)
plt.plot(t,xtilde[:,1],label=r"$\tilde{y}_\alpha$")
plt.plot(t,xtilde_NN[:,1],label=r"$\tilde{y}_{NN}$")
plt.xlabel("t")
plt.ylabel(r"$\tilde{\mathbf{y}}$")
plt.legend()
plt.grid()

plt.subplot(1,3,3)
plt.plot(t,xtilde[:,2],label=r"$\tilde{z}_\alpha$")
plt.plot(t,xtilde_NN[:,2],label=r"$\tilde{z}_{NN}$")
plt.xlabel("t")
plt.ylabel(r"$\tilde{\mathbf{z}}$")
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('tildes.png', format='png', dpi=300)
plt.show()

fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1.plot(*xtilde.T,label=r"$\alpha$")
ax1.set_xlabel(r"$\tilde{x}_\alpha$")
ax1.set_ylabel(r"$\tilde{y}_\alpha$")
ax1.set_zlabel(r"$\tilde{z}_\alpha$")
ax1.legend()
#plt.savefig('alpha-3d.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
plt.show()

fig2 = plt.figure(figsize=(7,7))
ax12 = fig2.add_subplot(1,1,1,projection='3d')
ax12.plot(*xtilde_NN.T,label=r"$NN$")
ax12.set_xlabel(r"$\tilde{x}_{NN}$")
ax12.set_ylabel(r"$\tilde{y}_{NN}$")
ax12.set_zlabel(r"$\tilde{z}_{NN}$")
#ax12.legend()
plt.savefig('nn-3d.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
plt.show()



# plt.figure(22,figsize=(12,4))
# plt.subplot(1,3,1)
# plt.plot(t,xh[:,0],label=r"${x}$")
# plt.plot(t,x_EM[:,0],label=r"${x}_{EM}$")
# plt.plot(t,x_NN[:,0],label=r"${x}_{NN}$")
# #plt.plot(t,xtilde[:,0],label=r"$\tilde{x}_{\alpha}$")
# plt.legend()
# plt.xlabel("t")
# plt.ylabel(r"${\mathbf{x}}$")
# plt.ylim(-20,20)
# plt.grid()

# plt.subplot(1,3,2)
# plt.plot(t,xh[:,1],label=r"${y}$")
# plt.plot(t,x_EM[:,1],label=r"${y}_{EM}$")
# plt.plot(t,x_NN[:,1],label=r"${y}_{NN}$")
# #plt.plot(t,xtilde[:,1],label=r"$\tilde{y}_{\alpha}$")
# plt.legend()
# plt.xlabel("t")
# plt.ylabel(r"${\mathbf{y}}$")
# plt.ylim(-25,25)
# plt.grid()

# plt.subplot(1,3,3)
# plt.plot(t,xh[:,2],label=r"${z}$")
# plt.plot(t,x_EM[:,2],label=r"${z}_{EM}$")
# plt.plot(t,x_NN[:,2],label=r"${z}_{NN}$")
# #plt.plot(t,xtilde[:,2],label=r"$\tilde{z}_{\alpha}$")
# plt.xlabel("t")
# plt.ylabel(r"${\mathbf{z}}$")
# plt.legend()
# plt.grid()
# plt.ylim(0,50)
# plt.tight_layout()
# plt.savefig('paths.png', format='png', dpi=300)
# plt.show()