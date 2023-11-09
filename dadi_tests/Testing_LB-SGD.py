import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import matplotlib.animation as animation
import LB_optimizer as LB
import time



def update_plot(ax, robot_pos, obstacle, robot_radius):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])

    # Plot the obstacle(s)
    for obs in obstacle:
        obs_circle = patches.Circle((obs[0], obs[1]), obs[2], color='red')
        ax.add_patch(obs_circle)

    # Plot the start, goal, and robot path
    ax.plot(*robot_start, 'go', markersize=10, label='Start')
    ax.plot(*robot_goal, 'bo', markersize=10, label='Goal')
    ax.plot(robot_pos[0], robot_pos[1], 'ro', markersize=robot_radius, label='Robot')
    line, = ax.plot([], [], 'b-', label='Robot Path')
    ax.legend()

d = 2
m = 2
experiments_num = 10
n = 20
n_iters = 4000
x00 = np.array([5, 5])
M0 = 0.5 / d
Ms = 0.0 * np.ones(m)
T = 3
sigma = 0.001
hat_sigma = 0.01
problem_name = 'QP'
L = 0.25    
obs_rad = 10
nu = 0.1 
eta0 = 0.5 
factor = 0.7
init_std = 0.5
delta = 0.01

 #Initialize LB-SGD optimizer
robot_start = ([5., 5.])
robot_goal = ([80., 80.])
obstacle = [(50., 50., 10.)]
sim_time = 30.
step_time = 0.1
N = int(sim_time / step_time)
current_time = 0.
robot_pos = np.array(robot_start)
# Constants for potential fields
k1 = 5 # Gain for goal attraction
k2 = 100  # Gain for obstacle repulsion



# Initialize the grid map
grid_size = (100, 100)
obstacle_cost = 1
obstacle_force = 2
fig, ax = plt.subplots()
G = nx.grid_2d_graph(*grid_size)
diagonal_cost = 0.5
robot_radius = 5


   # Define potential field functions f and h
def f(x):
    # Calculate the costs for each of the possible steps
    step_costs = [
        (x[0], x[1] + 1, 1),
        (x[0], x[1] - 1, 1),
        (x[0] + 1, x[1], 1),
        (x[0] - 1, x[1], 1),
        (x[0] + 1, x[1] + 1, diagonal_cost),
        (x[0] + 1, x[1] - 1, diagonal_cost),
        (x[0] - 1, x[1] + 1, diagonal_cost),
        (x[0] - 1, x[1] - 1, diagonal_cost)
    ]

    # Calculate the distance to the robot goal for each step and find the closest one
    closest_step = min(step_costs, key=lambda step: ((step[0] - robot_goal[0])**2 + (step[1] - robot_goal[1])**2)**0.5)

    repulsive_force = np.array([0., 0.])
    
    for obs in obstacle:
        diff = x - obs[:2]
        distance = np.linalg.norm(diff)
        
        
        if distance <= 2.4 * obs[2]:
            repulsive_force += k2 * ((2.4 * obs[2] - distance) / (distance**3)) * diff
 

        elif distance >= 2 * obs[2]:
            repulsive_force = np.array([0., 0.])   

    #print("repulsive_force",repulsive_force)     

    
    return  np.array([np.linalg.norm(closest_step[0] - robot_goal[0] + repulsive_force[0]), np.linalg.norm(closest_step[1] - robot_goal[1] + repulsive_force[1])]) 


def h(x):
    repulsive_force = np.array([0., 0.])
    
    for obs in obstacle:
        diff = x - obs[:2]
        distance = np.linalg.norm(diff)
        
        if distance <= 1.5 * obs[2]:
            repulsive_force -=   diff - np.array([2,2])
            #print("repulsive_force",repulsive_force)
        else:
            repulsive_force = np.array([0., 0.])

            
    return repulsive_force




def run_exp_LB_SGD(f, h, d, m,
                   experiments_num = experiments_num, 
                   n_iters = n_iters, 
                   n = n, 
                   M0 = M0, 
                   Ms = Ms, 
                   x00 = x00, 
                   sigma = sigma, 
                   nu = nu, 
                   eta0 = eta0, 
                   T = T, 
                   factor = factor, 
                   init_std = init_std,
                   delta = delta,
                   problem_name = problem_name):

    my_oracle = LB.Oracle(
        f = f,
        h = h, 
        sigma = sigma,
        hat_sigma = 0.01,
        delta = delta,
        m = m,
        d = d,
        nu = nu,
        zeroth_order = True,
        n = n)

    opt = LB.SafeLogBarrierOptimizer(
        x00 = x00,
        x0 = x00,
        M0 = M0,
        Ms = Ms,
        sigma = my_oracle.sigma,
        hat_sigma = my_oracle.hat_sigma,
        init_std = init_std,
        eta0 = eta0,
        oracle = my_oracle,
        f = f,
        h = h,
        d = d,
        m = m,
        reg = 0.1,
        factor = factor,
        T = T,
        K = int(n_iters / T / 2. / n),
        experiments_num = experiments_num,
        mu = 0.,
        convex = True,
        random_init = True,
        no_break = False)

    opt.run_average_experiment()


    for i in range(experiments_num):
        opt.constraints_total[i] = np.repeat(opt.constraints_total[i], 2 * n )
    
    constraints = opt.constraints_total
    runtimes = opt.runtimes
    runtimes = np.array(runtimes)
    
    """    with open('../runs/LB_SGD_' + problem_name + '_d' + str(d)  + '.npy', 'wb') as file:
        np.save(file, constraints)
        np.save(file, runtimes)
     """
    
    #print(opt)
    return opt
    
def update(self, x, obstacle):
        f_force = self.f(x)
        h_force = self.h(x, obstacle)
        total_force =  f_force + h_force
        #print("total_force",total_force)
        x = x - self.eta * total_force
        return x


def run_simulation(sim_time, step_time, robot_pos, robot_goal, robot_radius, obstacle, optimizer, update_plot):
    #current_time = 0
    #while current_time < sim_time:
    print("j length: ", len(plplp.x_total))
    for j in range(len(plplp.x_total)):
        print("j", j)
        #print("j", plplp.x_total[j]) 
        if j in [0,1]:
            for row in plplp.x_total[j]:
                
                total_force = row
                #print("exp_n", j,  "total_force", total_force)
                # Update the robot's position using the total force
                robot_pos = total_force
                update_plot(ax, robot_pos, obstacle, robot_radius)
                plt.pause(0.01)
               

        
        #current_time += step_time
    print("j length: ",len(plplp.x_total))
        
        
    return robot_pos



plplp= run_exp_LB_SGD(f, h, d, m,
                   experiments_num = experiments_num, 
                   n_iters = n_iters, 
                   n = n, 
                   M0 = M0, 
                   Ms = Ms, 
                   x00 = x00, 
                   sigma = sigma, 
                   nu = nu, 
                   eta0 = eta0, 
                   T = T, 
                   factor = factor, 
                   init_std = init_std,
                   delta = delta,
                   problem_name = problem_name)

#print("after_func",plplp)
# Then you can call the function like this:
robot_pos = run_simulation(sim_time, step_time, robot_pos, robot_goal, robot_radius, obstacle, plplp, update_plot)# Display the result
plt.show()