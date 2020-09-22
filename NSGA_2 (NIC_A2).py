# -*- coding: utf-8 -*-
import numpy as np
from math import pi
import matplotlib.pyplot as plt
no_of_cones = 1000
max_itern=50

def init_vars(no_of_cones):
    
    min_radius=0
    max_radius=10
    min_height = 0
    max_height = 20
    solution_r=np.random.rand(no_of_cones,1)*max_radius
    solution_h=np.random.rand(no_of_cones,1)*max_height
    #sol=np.array(solution_r,solution_h)
    sol=np.zeros((no_of_cones,2))
    #for i in range(len(solution_r)):
    #sol=(solution_r[i],solution_h[i]) for i in range(no_of_cones)
    i=0
    while i < len(solution_r):
        sol[i][0]=np.random.rand()*max_radius
        sol[i][1]=np.random.rand()*max_height
        t = check_vol(sol[i,:])
        if t==1:
            i = i+1
    #print('out_of_init')
    return sol    
def get_function_values(sol):
    function_values = np.zeros((len(sol),2))
    for i in range(len(sol)):
        function_values[i][0] = pi*(sol[i][0])*(np.sqrt((sol[i][0]**2)+(sol[i][1]**2)))
        function_values[i][1] = pi*(sol[i][0])*(sol[i][0]+(np.sqrt((sol[i][0]**2)+(sol[i][1]**2))))
    return function_values

def non_dom_sort(f1_f2):
    count=np.zeros((f1_f2.shape[0]))
    tmp1=f1_f2[:,0].copy()
    tmp2=f1_f2[:,1].copy()
    for k in range(len(f1_f2)):
        for l in range(len(f1_f2)):
            if (k!=l):
                #if (f1_f2[l,0]<=f1_f2[k,0] and f1_f2[l,1]<=f1_f2[k,1]) or (f1_f2[l,0]<f1_f2[k,0] or f1_f2[l,1]<f1_f2[k,1]):
                #    count[l]=count[l]+1
                if (f1_f2[l,0]>=f1_f2[k,0] and f1_f2[l,1]>=f1_f2[k,1]):
                    count[l]=count[l]+1
    return count
def crowding_dist(f1_f2):
    dist = np.zeros((f1_f2.shape[0]))
    dist[0] = 1e4
    dist[-1] = 1e4
    for k in range(len(f1_f2)):
        d1 = np.sort(f1_f2[:,0]) - f1_f2[k,0]
        d2 = np.sort(f1_f2[:,1])- f1_f2[k,1]
        #dist(k) = d1+d2
    return dist

def mutations_and_crossover(precent_of_crossover,sols):
    i=0
    while i<len(sols):
    #for i in range(len(sols)):
        sols[i,:]=sols[i,:]+2**(np.random.rand())
        sols[i,0] = sols[i,0]%10
        sols[i,1] = sols[i,1]%20
        t = check_vol(sols[i,:])
        if t==1:
            i=i+1
    #print(sols)
    number = int(precent_of_crossover*len(sols))
    op = np.zeros((number,2))
    for i in range(number):
        r = np.random.rand()
        indx1 = np.random.randint(0,len(sols))
        indx2 = np.random.randint(0,len(sols))
        op[i,:] = r * sols[indx1,:]+((1-r)*sols[indx2,:])
    return op  
def check_vol(arr):
    vol = pi*((arr[0])**2)*(arr[1])
    if vol>200:
        return 1
    else:
        return 0
    
sol = init_vars(no_of_cones) 
f1_f2 = np.zeros((no_of_cones,2))
f1_f2=get_function_values(sol)   
maxitern = 2
for i in range(maxitern):
    new_sol = mutations_and_crossover(1,sol)
    new_f1_f2 = get_function_values(new_sol)
    comb_f1_f2 = np.concatenate((f1_f2,new_f1_f2),axis = 0)
    non_dom_sort_order = non_dom_sort(comb_f1_f2)
    order = np.argsort(non_dom_sort_order)
    order = order[0:no_of_cones]
    f1_f2 = comb_f1_f2[order]
    sol = new_sol
fin_order = non_dom_sort(f1_f2)
#tmp_arg = np.where(fin_order=0)
t_arg = f1_f2[fin_order==0]
#t_pereto = f1_f2[t_arg,:]
plt.plot(f1_f2[:,0],f1_f2[:,1],'ro',markersize=3,label='other cones')   
plt.plot(t_arg[:,0],t_arg[:,1],'go', label='Pareto frontier')
plt.legend()
plt.xlabel('lat surface area')
plt.ylabel('total surface area')