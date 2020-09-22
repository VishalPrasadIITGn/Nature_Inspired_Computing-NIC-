# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:19:12 2020

@author: mtech2
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:36:47 2020

@author: vishal
"""
# %% importing libraries
import numpy as np
from math import pi,sin,cos
import math
import matplotlib.pyplot as plt
#%%initialize parameters
no_of_particle = 20
params = np.random.uniform(-1,1,(no_of_particle,22))
params[:,0:2]= params[:,0:2]*4
params[:,2:4] = abs(params[:,2:4])
params[:,4:6]= params[:,4:6]*4
params[:,6:8] = abs(params[:,6:8])
params[:,8:10]= params[:,8:10]*4
params[:,10:12] = abs(params[:,10:12])
params[:,12:14]= params[:,12:14]*4
params[:,14:16] = abs(params[:,14:16])

#no_of_particle
# %% breaking the parameters
mu_x1_y1 = params[:,0:2]
sigma_x1_y1 = params[:,2:4]
mu_x2_y2 = params[:,4:6]
sigma_x2_y2 = params[:,6:8]
mu_x3_y3 = params[:,8:10]
sigma_x3_y3 = params[:,10:12]
mu_x4_y4 = params[:,12:14]
sigma_x4_y4 = params[:,14:16]
# %% creating dataset
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
#xx, yy = np.meshgrid(x, y, sparse=True)
xy_label = np.zeros((10000,3))
count = 0
for i in range(len(x)):
    for j in range(len(y)):
        xy_label[count,0] = x[i]
        xy_label[count,1] = y[j]
        xy_label[count,2] = math.sin(pi * (x[i]/4))*math.sin(pi*(y[j]/4))
        count = count+1
#%%
def two_dim_gauss_val(val_arr, param_arr):
    x = val_arr[0]
    y = val_arr [1]
    mu_x = param_arr[0]
    mu_y = param_arr[1]
    sig_x = param_arr[2]
    sig_y = param_arr[3]
    a = np.exp(-np.power(x - mu_x, 2.) / ((2 *pi* np.power(sig_x, 2.)))**(1/2))
    b = np.exp(-np.power(y - mu_y, 2.) / ((2 *pi* np.power(sig_y, 2.)))**(1/2))
    return a*b
def anfis_fwd_pass(x_y,params):
#    mu_x1_y1 = params[:,0:2]
#    sigma_x1_y1 = params[:,2:4]
#    mu_x2_y2 = params[:,4:6]
#    sigma_x2_y2 = params[:,6:8]
#    mu_x3_y3 = params[:,8:10]
#    sigma_x3_y3 = params[:,10:12]
#    mu_x4_y4 = params[:,12:14]
#    sigma_x4_y4 = params[:,14:16]
    
    pred = np.zeros((x_y.shape[0]))
    for i in range(len(x_y)):    
    #for i in range(len(params)):
        a1 = two_dim_gauss_val(x_y[i,0:2],params[0:4])
        a2 = two_dim_gauss_val(x_y[i,0:2],params[4:8])
        b1 = two_dim_gauss_val(x_y[i,0:2],params[8:12])
        b2 = two_dim_gauss_val(x_y[i,0:2],params[12:16])
        
        pi1 = a1*b1
        pi2 = a2*b2
        
        pi1_norm = pi1 #/(pi1+pi2)
        pi2_norm = pi2 #/(pi1+pi2)
        
        # we can add more rules in below statements
        w1f1 = pi1_norm*((x_y[i,0]*params[16])+(x_y[i,1]*params[17])+params[18])
        w2f2 = pi2_norm*((x_y[i,0]*params[19])+(x_y[i,1]*params[20])+params[21])
        #print('1',w1f1,pi1_norm)
        #print('2',w2f2)
        pred[i] = w1f1+w2f2
    #print(x_y[:,2].shape,pred.shape)
    error = (np.sum((x_y[:,2]-pred)**2))/len(x_y)
    return pred,error
#anfis_fwd_pass(xy_label[0:10,0:2],params[0,:])
#def diff_evo(parameters):
maxiter = 20
final_pred_op = np.zeros((10000,maxiter))
parameters = params
for k in range(maxiter):
    print(k)
    # Performing mutation and crossover
    r1 = np.random.randint(0,no_of_particle,(no_of_particle,1))
    r2 = np.random.randint(0,no_of_particle,(no_of_particle,1))
    r3 = np.random.randint(0,no_of_particle,(no_of_particle,1))
    for i in range(len(r1)):
        if r1[i]==i:
            r1[i] = r1[i-1]
        if r2[i]==i:
            r2[i] = r2[i-1]
        if r3[i]==i:
            r3[i] = r3[i-1]
    trial_vec = parameters[r1] + 0.2*(parameters[r2] - parameters[r3])
    trial_vec = np.reshape(trial_vec,(no_of_particle,22))
    for r in range(trial_vec.shape[0]):
        tem = np.random.randint(0,trial_vec.shape[1])
        trial_vec[i,0:tem] = parameters[i,0:tem]
    predicted = np.zeros((10000,no_of_particle))
    predicted_trial = np.zeros((10000,no_of_particle))
    err = np.zeros((len(params)))
    err_trial = np.zeros((len(err)))
    #    predicted[:,j],err[j] = anfis_fwd_pass(xy_label,params[j,:])
    
    for j in range(len(params)):
        predicted[:,j],err[j] = anfis_fwd_pass(xy_label,params[j,:])
    for j in range(len(trial_vec)):
        predicted_trial[:,j],err_trial[j] = anfis_fwd_pass(xy_label,trial_vec[j,:])
    idx = err>err_trial
    params[idx] = trial_vec[idx]
    min_err = np.argmin(err)
    #final_pred_op[:,k] = predicted[:,min_err]
    final_pred_op[:,k] = np.mean(predicted, axis=-1)
np_meshgrid = np.reshape(final_pred_op,(100,100,maxiter))

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
#Z = f(X, Y)
#Z = np_meshgrid_res
#for t in range(maxiter):
#    Z =  np_meshgrid[:,:,t]
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    ax.contour3D(X, Y, Z, 50, cmap='binary')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_zlabel('z');

np_meshgrid_res = np.reshape(xy_label[:,2],(100,100))
Z = np_meshgrid_res
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z');

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Original results')
#ax.set_xlabel('x ')
ax.set_ylabel('y')
plt.show()

import imageio
images = []
for t in range(maxiter):
    Z =  np_meshgrid[:,:,t]
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Values using ANFIS (iter='+str(t)+')')
    #ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(str(t)+'.png')
    images.append(imageio.imread(str(t)+'.png'))
    plt.show()
imageio.mimsave('movie.gif', images, duration = 0.5)
#import imageio
#images = []
#for filename in filenames:
#    images.append(imageio.imread(filename))
#imageio.mimsave('/path/to/movie.gif', images)

#x_y = xy_label
#params = params[0,:]
#for j in range(1):
#    pred = np.zeros((x_y.shape[0]))
#    for i in range(len(x_y)):    
#    #for i in range(len(params)):
#        a1 = two_dim_gauss_val(x_y[i,0:2],params[0:4])
#        a2 = two_dim_gauss_val(x_y[i,0:2],params[4:8])
#        b1 = two_dim_gauss_val(x_y[i,0:2],params[8:12])
#        b2 = two_dim_gauss_val(x_y[i,0:2],params[12:16])
#        
#        pi1 = a1*b1
#        pi2 = a2*b2
#        
#        pi1_norm = pi1/(pi1+pi2)
#        pi2_norm = pi2/(pi1+pi2)
#        # we can add more rules in below statements
#        w1f1 = pi1_norm*(x_y[i,0]*params[16]+x_y[i,1]*params[17]+params[18])
#        w2f2 = pi2_norm*(x_y[i,0]*params[19]+x_y[i,1]*params[20]+params[21])
#        #print('1',w1f1,pi1_norm)
#        #print('2',w2f2)
#        pred[i] = w1f1+w2f2
#    print(x_y[:,2].shape,pred.shape)
#    error = (np.sum((x_y[:,2]-pred)**2))/len(x_y)        