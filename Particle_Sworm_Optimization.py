import random
import numpy as np 
from math import pi,sin,cos,exp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
#function that models the problem
def cost(position):
    expcomponent = abs(100 - (np.sqrt(position[0]**2 + position[1]**2) / pi));
    scores = -0.0001 * ((abs(sin(position[0]) * sin(position[1]) * exp(expcomponent)) + 1) ** 0.1)
    #return position[0]**2 + position[1]**2 + 1
    return scores
#Some variables to calculate the velocity
inertia_coff = 0.6
c1 = 0.5
c2 = 0.5
max_iter = 100
delta = 1e-17
no_of_particles = 50
cordinates_particle = (np.random.randn(no_of_particles,2)-0.5)*10
particle_pbest_cordinates = cordinates_particle
particle_pbest_cost = np.full((no_of_particles), 1e15, dtype=float)
global_best_cost = 1e15 #float('inf')
prev_gbest_cost = 1e15 #float('inf')
gbest_position = np.array([1e15, 1e15])
velocity_vector = np.zeros((no_of_particles,2))
iteration = 0
x_for_all=np.zeros((max_iter,no_of_particles))
y_for_all=np.zeros((max_iter,no_of_particles))
vel_x_for_all=np.zeros((max_iter,no_of_particles))
vel_y_for_all=np.zeros((max_iter,no_of_particles))
gbest=np.zeros((max_iter,2))
gbest_cost=np.zeros((max_iter,1))
#while iteration < max_iter:
for iteration in range(max_iter):    
    print('Gbest cordinates for teration ',iteration,' is ', gbest_position, 'and cost is', global_best_cost)
    prev_cordinates_particle = np.copy(cordinates_particle)
    temp = np.zeros((no_of_particles,1))
    for i in range(no_of_particles):
        cost_particle = cost(cordinates_particle[i])
        temp[i] = cost_particle 
        #print(iteration,cost_particle, 'iteration,cost_particle cordinates_particle[i]', cordinates_particle[i])
        
        if(cost_particle < particle_pbest_cost[i]):
            particle_pbest_cost[i] = cost_particle
            particle_pbest_cordinates[i] = cordinates_particle[i]

        if(cost_particle<global_best_cost):
            prev_gbest_cost = global_best_cost
            global_best_cost = cost_particle
            gbest_position = cordinates_particle[i]
        
        k1=c1*np.random.rand()
        k2=c2*np.random.rand()
        term1 = velocity_vector[i]
        term2 = particle_pbest_cordinates[i] - cordinates_particle[i]
        term3 = gbest_position-cordinates_particle[i]
        new_velocity = (inertia_coff*term1) + (k1 * term2) + (k2 * term3)
        cordinates_particle_new = new_velocity + cordinates_particle[i]
        if (np.abs(cordinates_particle_new[0])<10 and np.abs(cordinates_particle_new[1])<10):
            cordinates_particle[i] = cordinates_particle_new
        vel_x_for_all[iteration][i]=new_velocity[0]
        vel_y_for_all[iteration][i]=new_velocity[1]
    temp=np.around(temp,4)
    flag=0
    for r in range(len(temp)):
        if temp[0]!=temp[r]:
            flag=1
            break
    if(abs(global_best_cost - prev_gbest_cost) < delta or flag==0):
        print(abs(global_best_cost - prev_gbest_cost))
        break
    #cordinates_particle = np.clip(cordinates_particle,-10,10)
    #for i in 
    x_for_all[iteration]=cordinates_particle[:,0]
    y_for_all[iteration]=cordinates_particle[:,1]
    gbest[iteration]=gbest_position
    gbest_cost[iteration]=global_best_cost


x = np.linspace(-9, 9,190) 
y = np.linspace(-9, 9,190) 
x_1, y_1 = np.meshgrid(x, y)
#map(cost, (x_1,y_1))
x_1_flat=np.ndarray.flatten(x_1)
y_1_flat=np.ndarray.flatten(y_1)
cost_flat=np.zeros((len(x_1_flat),1))
for i in range(len(x_1_flat)):
    cost_flat[i] = cost([x_1_flat[i],y_1_flat[i]])   
cost_reshaped = cost_flat.reshape(x.shape[0],x.shape[0])
print("The best cordinates are ", gbest_position, 'and cost is', global_best_cost)
fig, ax = plt.subplots(figsize=(20, 10))
points, = ax.plot(np.random.rand(no_of_particles), 'go', ms=12, label='other_particles')
points2, = ax.plot(np.random.rand(1), 'ro', ms=12, label='gbest(cost)_particle')
title = ax.text(0.5,0.95, "",fontsize=24, bbox={'facecolor':'W', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
plt.contourf(x_1, y_1, cost_reshaped, cmap = 'jet') 
plt.colorbar() 
ax.grid()
#patch = patches.Arrow(x_for_all[0], y_for_all[0], vel_x_for_all[0], vel_y_for_all[0] )
#patch = plt.Arrow(0.1,0.1,0.1,0.1 )
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
def update(data):
    
    points.set_xdata(data[0])
    points.set_ydata(data[1])
    title.set_text(u"Itern: {}, gbest_cost={} at{}".format(data[2],data[3],data[4]))
    points2.set_xdata(data[4][0])
    points2.set_ydata(data[4][1])
     
    return points, points2

def generate_points():
    for i in range(iteration):
        yield (x_for_all[i],y_for_all[i],i,gbest_cost[i],gbest[i], vel_x_for_all[i], vel_y_for_all[i])  # change this

ani = animation.FuncAnimation(fig, update, generate_points, interval=2000, repeat_delay=50000)
ani.save('animation3.gif', writer='imagemagick', fps=2);
plt.show()
fig = plt.figure(figsize=(20, 10))
plt.plot(gbest_cost)
plt.ylabel('fitness/cost value')
plt.xlabel('no of iteration')
plt.title('Cost vs iteration')
fig.savefig('cost vs itern plot.png')