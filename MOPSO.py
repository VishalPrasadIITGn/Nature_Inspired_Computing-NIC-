# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:44:50 2020
NIC A3 MOPSO
@author: mtech2
"""
# %% importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# %% reading xlsx file
#df = pd.ExcelFile("Stock_information.xlsx")
dfs = pd.read_excel("Stock_information.xlsx", header = None)
dfs2 = pd.DataFrame(dfs)

return_val = dfs2.iloc[9,1:]
return_val_np = np.zeros((return_val.shape))
return_val_np[:] = return_val.iloc[:]

var_sigma_sq = dfs2.iloc[10,1:]
var_sigma_sq_np = np.zeros((var_sigma_sq.shape))
var_sigma_sq_np[:] = var_sigma_sq.iloc[:]

corr_matrix = dfs2.iloc[17:36,1:]
corr_matrix_np = np.zeros((19,20))
corr_matrix_np[:,:]= corr_matrix.iloc[:,:]

std_dev_np = np.sqrt(var_sigma_sq_np)

def check_params(arr_params):
    if (np.sum(arr_params[:]<=0.15)==20) and np.sum(arr_params[0:4])<=0.4 and np.sum(arr_params[4:8])<=0.4 and np.sum(arr_params[8:12])<=0.4 and np.sum(arr_params[12:16])<=0.4 and np.sum(arr_params[16:19])<=0.4:
        return 1
    else:
        return 0
def calculate_risk(params, var_sigma_sq_np, corr_matrix_np):
    std_dev_np = np.sqrt(var_sigma_sq_np)
    params_sq = np.square(params)
    #no_of_particle = arr_params_all.shape[0]
    no_of_features = params.shape[0]
    prod_matrix = np.zeros((num_particles))
    risk = np.dot(params_sq,var_sigma_sq_np)
    for i in range(no_of_features-1):
        for j in range(i+1,no_of_features):
            risk = risk+(2*params[i]*params[j]*std_dev_np[i]*std_dev_np[j]*corr_matrix_np[i,j])
            #temp2 = params[i,:]
            #temp2 = np.sqrt(var_sigma_sq_np[i,:])
    return np.sqrt(risk)
def non_dom_sort(f1_f2):
    count=np.zeros((f1_f2.shape[0]))
    tmp1=f1_f2[:,0].copy()
    tmp2=f1_f2[:,1].copy()
    for k in range(len(f1_f2)):
        for l in range(len(f1_f2)):
            if (k!=l):
                #if (f1_f2[l,0]<=f1_f2[k,0] and f1_f2[l,1]<=f1_f2[k,1]) or (f1_f2[l,0]<f1_f2[k,0] or f1_f2[l,1]<f1_f2[k,1]):
                #    count[l]=count[l]+1
                if (f1_f2[k,0]<=f1_f2[l,0] and f1_f2[k,1]>=f1_f2[l,1]):
                    count[k]=count[k]+1
    #print(count.shape)
    return count
 # %% initialising varialbles.
 
maxiter = 50
num_particles = 500    
params = np.random.rand(num_particles,20)%0.15
risk_all = np.zeros((maxiter,num_particles))
return_all = np.zeros((maxiter,num_particles))
risk_gbest = np.zeros((maxiter,num_particles))
pbest_params = params
pbest_cost = np.zeros((num_particles,2))
pbest_cost[:,0] = -1e15
pbest_cost[:,1] = 1e15
gbest_cost = np.array([-1e15,1e15])
gbest_params = np.zeros((20))
inertia_coff = 0.2
c1 = 0.02
c2 = 0.02
velocity_old = np.random.rand(num_particles,20)%0.10
new_velocity = np.zeros((params.shape))
for itern in range(maxiter):
    temp = np.zeros((num_particles,2))
    prev_params = np.copy(params)
    for i in range(num_particles):
        retrn =  np.dot(params[i,:],return_val_np)
        risk = calculate_risk(params[i,:],var_sigma_sq_np,corr_matrix_np)
        return_all[itern,i] = retrn
        risk_all[itern,i] = risk
        temp[0] = retrn
        temp[1] = risk
        if pbest_cost[i,0]<=retrn and pbest_cost[i,1]>=risk:
            pbest_cost[i,0] = retrn
            pbest_cost[i,1] = risk
            pbest_params[i,:] = params[i,:]
        if gbest_cost[0]<=retrn and gbest_cost[1]>=risk:
            gbest_cost[0] = retrn
            gbest_cost[1] = risk
            gbest_params = params[i,:]
        
        k1=c1*np.random.rand()
        k2=c2*np.random.rand()
        #term1 = velocity_vector[i]
        term2 = pbest_params[i,:] - params[i,:]
        term3 = gbest_params-params[i,:]
        new_velocity[i,:] =  (0.2*velocity_old[i,:])+(k1 * term2) + (k2 * term3)
        temp_params = new_velocity[i,:] + params[i,:]
        #velocity_old[i,:] = new_velocity
        qw = check_params(temp_params)
        if qw==1:
            params[i,:] = temp_params
        else:
            params[i,:] = temp_params%0.15
        #if (np.abs(cordinates_particle_new[0])<10 and np.abs(cordinates_particle_new[1])<10):
        #    cordinates_particle[i] = cordinates_particle_new
    velocity_old = new_velocity
count = np.zeros((maxiter,num_particles))
for i in range(maxiter):
    arr1 = return_all[i,:]
    arr2 = risk_all[i,:]
    ret_risk_arr = np.array([arr1,arr2]).T
    count[i,:] = non_dom_sort(ret_risk_arr)
risk_max = np.nanmax(risk_all)
ret_max = np.nanmax(return_all)
import imageio
images = []
for i in range(maxiter):
    #print(i)
    temp_count = count[i,:].T
    temp_ret = return_all[i,:]
    temp_risk = risk_all[i,:]
    temp_ret_p = temp_ret[temp_count==0]
    temp_risk_p = temp_risk[temp_count==0]
    #plt.figure()
    plt.xlim(0,ret_max)
    plt.ylim(0,risk_max)
    plt.plot(temp_ret,temp_risk,'ro',markersize=3,label='other cones')   
    plt.plot(temp_ret_p,temp_risk_p,'go', label='Pareto frontier')
    plt.ylabel('Risk(portfolio std dev)')
    plt.xlabel('Return(wtd)')
    plt.title('itern:'+str(i))
    plt.savefig(str(i)+'.png')
    images.append(imageio.imread(str(i)+'.png'))
    plt.show()
imageio.mimsave('movie_and.gif', images, duration = 0.3)