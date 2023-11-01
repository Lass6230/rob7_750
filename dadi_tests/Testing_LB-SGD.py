import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import matplotlib.animation as animation
import LB_optimizer as LB

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
n = int(d / 2)
n = 1
n_iters = d * 60
x_opt = np.ones(d) / d**0.5
x00 = np.zeros(d)
M0 = 0.5 / d
Ms = 0.0 * np.ones(m)
T = 3
sigma = 0.001
problem_name = 'QP'
L = 0.25    
obs_rad = 10

 #Initialize LB-SGD optimizer
robot_start = (5., 5.)
robot_goal = (90., 90.)
obstacle = [(50., 50., 10.)]
sim_time = 30.
step_time = 0.1
N = int(sim_time / step_time)
current_time = 0.
robot_pos = np.array(robot_start)
# Constants for potential fields
k1 = 5.0  # Gain for goal attraction
k2 = 2.0  # Gain for obstacle repulsion
step_size = 1.0  # Control the step size


# Initialize the grid map
grid_size = (100, 100)
obstacle_cost = 1
#obstacle_force = 2
fig, ax = plt.subplots()
G = nx.grid_2d_graph(*grid_size)
diagonal_cost = 1
robot_radius = 5

moves = [(0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1), (1, 1, diagonal_cost), (1, -1, diagonal_cost), (-1, 1, diagonal_cost), (-1, -1, diagonal_cost)]

   # Define potential field functions f and h
def f(x, robot_goal=robot_goal):
    return -k1 * (x - robot_goal)


def h(x):
    obstacle_force = np.zeros(2)
    for obs in obstacle:
        diff = x - obs[:2]
        if np.linalg.norm(diff) <= obs[2]:
            obstacle_force += k2 * (1 / np.linalg.norm(diff) - 1 / obs[2]) * diff
            
    return obstacle_force

"""def h(x):
    d = np.size(x)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return A.dot(x) - b 

def f(x):    
    d = np.size(x)
    xx = 2.0 * np.ones(d)
    return np.linalg.norm(x - xx, 2)**2 / 4.0 / d"""



def run_exp_LB_SGD(f, h, d, m,
                   experiments_num = 5, 
                   n_iters = 100, 
                   n = 1, 
                   M0 = 0.5 / 2., 
                   Ms = 0. * np.ones(4), 
                   x00 = np.zeros(2), 
                   x_opt =  np.ones(2) / 2**0.5, 
                   sigma = 0.001, nu = 0.01, 
                   eta0 = 0.05, 
                   T = 3, 
                   factor = 0.85, 
                   init_std = 0.1,
                   delta = 0.01,
                   problem_name = ''):

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
        reg = 0.0001,
        x_opt = x_opt,
        factor = factor,
        T = T,
        K = int(n_iters / T / 2. / n),
        experiments_num = experiments_num,
        mu = 0.,
        convex = True,
        random_init = True,
        no_break = True)

    opt.run_average_experiment()
    
    for i in range(experiments_num):
        opt.errors_total[i] = np.repeat(opt.errors_total[i], 2 * n)
        opt.constraints_total[i] = np.repeat(opt.constraints_total[i], 2 * n )
    
    errors = opt.errors_total
    constraints = opt.constraints_total
    runtimes = opt.runtimes
    runtimes = np.array(runtimes)
    
    """    with open('../runs/LB_SGD_' + problem_name + '_d' + str(d)  + '.npy', 'wb') as file:
        np.save(file, errors)
        np.save(file, constraints)
        np.save(file, runtimes)
     """
    return opt
    
def update(self, x, obstacle):
        f_force = self.f(x, self.x_opt)
        h_force = self.h(x, obstacle)
        total_force =  f_force + h_force
        x = x - self.eta * total_force
        return x


def run_simulation(sim_time, step_time, robot_pos, robot_goal, robot_radius, obstacle, optimizer, update_plot):
    current_time = 0
    while current_time < sim_time:
        # Calculate the total force on the robot
        total_force = x_opt 
        print("Current time: ", total_force)
        # Update the robot's position using the total force
        robot_pos = robot_pos + step_size * total_force
        
        # Check if the robot has reached the goal
        if np.linalg.norm(robot_pos - robot_goal) < robot_radius:
            print("Robot reached the goal!")
            break
        
        current_time += step_time

        update_plot(ax, robot_pos, obstacle, robot_radius)
        plt.pause(0.01)  # Add a slight delay to see the animation
    return robot_pos



plplp= run_exp_LB_SGD(f, h, d, m,
               experiments_num = experiments_num, 
               n_iters = n_iters,
               n = n, 
               M0 = M0, 
               Ms = Ms, 
               x00 = x00, 
               init_std = 0.,
               eta0 = 0.1,
               delta = 0.01,
               nu = 0.01, 
               factor = 0.7,
               T = 7,
               x_opt = x_opt, 
               sigma = sigma,
               problem_name = problem_name)


# Then you can call the function like this:
robot_pos = run_simulation(sim_time, step_time, robot_pos, robot_goal, robot_radius, obstacle, plplp, update_plot)# Display the result
plt.show()