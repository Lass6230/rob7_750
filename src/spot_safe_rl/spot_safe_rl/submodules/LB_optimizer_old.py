# Write the documentation

import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp, conelp, coneqp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import matplotlib.animation as animation
import math

# import matplotlib.pyplot as plt
# import matplotlib.lines as line
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
import numdifftools as nd
from time import time

from typing import Callable
from dataclasses import dataclass


@dataclass
class Oracle:    
    """
    This class allows to sample from the first-order noisy oracle given the objective f and constraints h. 
    
    Given the functions and noise parameters, it samples:
                stochastic value and gradient of objective f: objective_grad/values, 
                stochastic value and gradient of constraints h: constraints_grad/values
                alphas: 
    
    It can be zeroth-order oracle when "zeroth_order: true", 
        in this case it estimates the stochastic gradient using the finite difference and s ~ U(S(0,1))
    
    
    Parameters:
        f: Callable, objective
        h: Callable, vector of constraint
        df: np.array, objective gradient
        dh: np.array, constraint gradient
        sigma: float, variance of the Gaussian value noise  
        hat_sigma: float, variance of the Gaussian gradient noise (in the first-order oracle case)
        delta: float, confidence level
        m: int, number of constraints
        d: int, dimensionality
        nu: float, sampling radius (in the zeroth-order oracle case)
        objective_value: float, stochastic oracle output: objective value
        constraints_values: np.array, stochastic oracle output: constraint values, dimensionality m
        alphas: np.array, lower confidence bounds on alphas [-f^i(x)]
        objective_grad: np.array, stochastic oracle output: objective gradient
        constraints_grad: np.array, stochastic oracle output: constraint gradients
        zeroth_order: bool, zeroth-order or first-order initial information
        n: int, number of s-samples per iteration 
        
    """
    f: Callable = None                  
    h: Callable = None                  
    df: np.array = None                 
    dh: np.array = None                 
    sigma: float = None                 
    hat_sigma: float = None             
    delta: float = None                 
    m: int = None                      
    d: int = None                      
    nu: float = None                    
    objective_value: float = None       
    constraints_values: np.array = None 
    alphas: np.array = None             
    objective_grad: np.array = None     
    constraints_grad: np.array = None   
    zeroth_order: bool = True           
    n: int = 1                          
        
    def sample(self, x: np.array) -> None:
        self.objective_value = self.f(x) + np.random.normal(0, self.sigma / self.n**0.5)
        self.constraints_values = self.h(x) + np.random.normal(0, self.sigma / self.n**0.5, self.m)
        self.df = nd.Gradient(self.f)(x)
        self.dh = nd.Gradient(self.h)(x)
        if self.zeroth_order:
            self.hat_sigma = self.d * (self.sigma / self.nu + self.nu)
            for j in range(self.n):
                s_unnormalized = np.random.normal(0, 1, self.d)
                s = s_unnormalized / np.linalg.norm(s_unnormalized)
                if j == 0:
                    self.objective_grad = (self.d *
                                       (self.f(x + self.nu * s) 
                                        + np.random.normal(0, self.sigma) - self.objective_value)
                                        / self.nu) * s / self.n
                    self.constraints_grad = (np.outer((self.d *
                                        (self.h(x + self.nu * s) + np.random.normal(0, self.sigma, self.m) -
                                        self.constraints_values) / self.nu), s)) / self.n
                else:
                    self.objective_grad += (self.d *
                                           (self.f(x + self.nu * s) 
                                            + np.random.normal(0, self.sigma) - self.objective_value)
                                           / self.nu) * s / self.n
                    self.constraints_grad += (np.outer((self.d *
                                            (self.h(x + self.nu * s) + np.random.normal(0, self.sigma, self.m) -
                                            self.constraints_values) / self.nu), s)) / self.n
                self.alphas = - self.constraints_values -\
                    (np.log(1. / self.delta))**0.5 * self.sigma / self.n**0.5 * np.ones(self.m) - self.nu * np.ones(self.m)
        else:
            self.objective_grad = self.df + np.random.normal(0, self.hat_sigma / self.n**0.5, self.d)
            #print("objective_grad from oracle", self.objective_grad)
            self.constraints_grad = self.dh + np.random.normal(0, self.hat_sigma / self.n**0.5, (self.m, self.d))
            #print("constraints_grad from oracle", self.constraints_grad)
            self.alphas = - self.constraints_values - \
                (np.log(1. / self.delta))**0.5 * self.sigma / self.n**0.5 / 2. * np.ones(self.m)
            #print("alphas from oracle", self.alphas)


@dataclass
class SafeLogBarrierOptimizer:
    """
    This class allows to run LB-SGD optimization procedure given the oracle for the objective f and constraint h. 
    """
    x_last = None
    obstacle_list: list = None
    obstacle: np.array = None
    obs_pos_x: float = 50.0
    obs_pos_y: float = 50.0
    obs_pos_z: float = 10.0
    x00: np.array = None
    x0: np.array = None
    M0: float = None
    Ms: np.array = None
    sigma: float = None
    hat_sigma: float = None
    init_std: float = 0. 
    eta0: float = None
    eta: float = None
    step: np.array = None
    oracle: Oracle = None
    f: Callable = None
    h: Callable = None
    d: float = None
    m: float = None
    reg: float = None
    # x_opt: float = None
    T: int = None
    K: int = None
    experiments_num: int = None
    mu: float = None
    xs: list = None
    s: int = None
    convex: bool = None
    random_init: bool = False
    no_break: bool = True
    x_total: list = None
    errors_total: list = None
    constraints_total: list = None
    beta: float = None
    factor: float = 0.5
    runtimes: list = None
    t_count: int = 0
    
    def compute_gamma(self, t: int) -> float:
        """
        Computes the step-size
        
        Args:
            t: int, iteration number, not used
        """
        step_norm = np.linalg.norm(self.step)
        alphas = self.oracle.alphas
        dhs = self.oracle.constraints_grad
        
        alphas_reg = alphas
        L_dirs = np.zeros(self.m)
        for i in range(self.m):
            L_dirs[i] = np.abs((dhs[i].dot(self.step)) / step_norm) +\
                (np.log(1. / self.oracle.delta))**0.5 * self.hat_sigma / self.oracle.n**0.5 
            alphas_reg[i] = max(self.reg, alphas[i])

        M2 = self.M0 + 2 * self.eta * np.sum(self.Ms / alphas_reg) + 4 * self.eta * np.sum(L_dirs**2 / alphas_reg**2) 
        #print("M2", M2)
        gamma = min(1. / step_norm * np.min(alphas / ( 2 * L_dirs +  alphas_reg**0.5 * self.Ms**0.5)), 
                    1. / M2 )
        
        #print("gamma", gamma)
        return gamma

    def dB_estimator(self):
        """
        Computes the log barrier gradient estimator
        """
        alphas = self.oracle.alphas
        jacobian = self.oracle.constraints_grad
        #print("jacobian", jacobian)
        df_e = self.oracle.objective_grad
        #print("df_e", df_e)
        denominators = 1. / np.maximum(np.ones(self.m) * self.reg, alphas)
        #print("denominators", denominators)
        dB = df_e + self.eta * jacobian.T.dot(denominators)
        #print("eta", self.eta)
        return dB
    
    def barrier_SGD(self):
        """
        Runs LB_SGD with constant parameter eta
        """
        self.xs = []
        # x_trajectory = [] # added
        xt = self.x0
        #print("im,xt",xt)
        Tk = 0    
        for t in range(self.T):
            #print("ORacle Xo", xt)
            self.oracle.sample(xt)  
            
            self.step = self.dB_estimator()
            step_norm = np.linalg.norm(self.step)
            #print("step_norm", step_norm)
            gamma = self.compute_gamma(t)

            if step_norm < self.eta and self.no_break == False:
                break

            xt = xt - gamma * self.step # calculate and update policy
            #print("xt", xt)
            Tk += 1
            if t == 0:
                x_trajectory = np.array([xt]) # # is the policy
                gamma_trajectory = np.array([gamma])
                
                constraints_trajectory = np.max(self.h(xt))
                worst_constraint = np.max(self.h(xt))
            else:
                x_trajectory = np.vstack((x_trajectory, xt)) # is the policy
                gamma_trajectory = np.vstack((gamma_trajectory, gamma))
                
                constraints_trajectory = np.hstack((constraints_trajectory, np.max(self.h(xt))))
                worst_constraint = max(worst_constraint, np.max(self.h(xt)))
            
            # print("x_trajectory", x_trajectory)
            #print("gammea_trajectory", gamma_trajectory)
            
            #print("constraints_trajectory", constraints_trajectory)
            #print("worst_constraint", worst_constraint)    

            self.xs.append(xt)
            x_last = xt
    
        return x_trajectory, gamma_trajectory,  constraints_trajectory, x_last, Tk
          
    def log_barrier_decaying_eta(self):
        """
        Outer loop of LB-SGD with decreasing eta


        """
        x_obstacle_trajectory = self.obstacle
        
        x_long_trajectory = self.x0
        
        constraints_long_trajectory = np.max(self.h(self.x0))    
        T_total = 0
        
        self.eta = self.eta0
        x0 = self.x0
        x_prev = x0
        # print(self.f(self.x0))
          
        print("x_long_trajectory in log_barrier_decay", x0)
       
        print("constraints_long_trajectory in log_barrier_decay", constraints_long_trajectory)
        print("T_total in log_barrier_decay", T_total)
        
        
        for k in range(self.K):
                
            x_traj_k, gamma_traj_k, constraints_traj_k, x_last_k, T_k = self.barrier_SGD()
            
            constraints_long_trajectory = np.hstack((constraints_long_trajectory, constraints_traj_k))
            x_long_trajectory = np.vstack((x_long_trajectory, x_traj_k))
            T_total = T_total + T_k
            self.x0 = x_last_k
            self.eta = self.eta * self.factor
            self.obs_pos_x -= 1.
            self.obs_pos_y -= 1.01
            self.obstacle = [(self.obs_pos_x,self.obs_pos_y,self.obs_pos_z)]
            x_obstacle_trajectory = np.vstack((x_obstacle_trajectory,self.obstacle))
            
            #print("eta in LB",self.eta)
            """"            print("x_traj_k", x_traj_k)
            print("gamma_traj_k", gamma_traj_k) 
            print("errors_traj_k", errors_traj_k)
            print("constraints_traj_k", constraints_traj_k)
            print("x_last_k", x_last_k)
            print("T_k", T_k)
            """
        return x_obstacle_trajectory, x_long_trajectory,  constraints_long_trajectory, T_total, x_last_k 

    def get_random_initial_point(self):
        """
        Obtains random safe initial point
        """
        x0_det = self.x00
        d = self.d
        
        
        for i in range(1000 * d):
            x0 =  x0_det + np.random.uniform(low=-1, high=1, size=self.d) * self.init_std
            #print(self.h(x0))
            if (self.h(x0) < - self.beta).all():
                break
        return x0
    


    def run_previous_model(self):
        print("online traning on previouss online traninged mode stuff :P")
        
        # load np.array, that are our "model"
        pre_xt = np.load("runs.npy")
        
        #print(self.d)
        self.beta = self.eta0
        if self.random_init:
            self.x0 = self.get_random_initial_point()
            #self.x0 = pre_xt[0][len(pre_xt)-1] ## added to hopfully start at where the previous online training stopped
            self.x0 = pre_xt[0]
        else:
            self.x0 = self.x00
        f_0 = self.f(self.x0[0])
        
        
        time_0 = time() 
        (x_obstacle_trajectory, x_long_trajectory, constraints_long_trajectory, 
                            T_total, 
                            x_last) = self.log_barrier_decaying_eta()
        self.runtimes = [time() - time_0]
        
        x_total = []
        x_obstacle = []
        errors_total = []
        constraints_total = []

        
        constraints_total.append(constraints_long_trajectory)
        #print("HEy, DADi LOOK HERERE",self.x0)
        for i in range(self.experiments_num - 1):
            if self.random_init:
                self.x0 = self.get_random_initial_point()
                self.x0 = pre_xt[0]
                f_0 = self.f(self.x0[0])
                ## added to hopfully start at where the previous online training stopped
            else:
                self.x0 = self.x00
            

            time_0 = time() 
            (x_obstacle_trajectory, x_obstacle_trajectory,x_long_trajectory, constraints_long_trajectory, 
                                T_total, 
                                x_last) = self.log_barrier_decaying_eta()
            self.runtimes.append(time() - time_0)
            x_total.append(x_long_trajectory)
            x_obstacle.append(x_obstacle_trajectory)
            constraints_total.append(constraints_long_trajectory)
        self.x_total = x_total
        self.obstacle_list = x_obstacle
        #print(x_total)
        self.constraints_total = constraints_total
        print('LB_SGD runs finished')

        return x_last
    
    def run_average_experiment(self):
        """
        Runs the LB_SGD multiple times, 
        
        Outputs: x_last, 
        Updates: errors_total, constraints_total, xs

        
        
        """

        #print(self.d)
        self.beta = self.eta0
        if self.random_init:
            self.x0 = self.get_random_initial_point()
        else:
            self.x0 = self.x00
        f_0 = self.f(self.x0)
        
        time_0 = time() 
        (x_obstacle_trajectory, x_long_trajectory, constraints_long_trajectory, 
                            T_total, 
                            x_last) = self.log_barrier_decaying_eta()
        self.runtimes = [time() - time_0]
        
        x_total = []
        x_obstacle = []
        errors_total = []
        constraints_total = []

        
        constraints_total.append(constraints_long_trajectory)
        #print("HEy, DADi LOOK HERERE",self.x0)
        for i in range(self.experiments_num - 1):
            if self.random_init:
                self.x0 = self.get_random_initial_point()
                f_0 = self.f(self.x0)
            else:
                self.x0 = self.x00

            time_0 = time() 
            (x_obstacle_trajectory,x_long_trajectory, constraints_long_trajectory, 
                                T_total, 
                                x_last) = self.log_barrier_decaying_eta()
            self.runtimes.append(time() - time_0)
            x_total.append(x_long_trajectory)
            x_obstacle.append(x_obstacle_trajectory)
            
            constraints_total.append(constraints_long_trajectory)
        self.x_total = x_total
        self.obstacle_list = x_obstacle
        #print(x_total)
        self.constraints_total = constraints_total
        print('LB_SGD runs finished')

        return x_last
    

    
    def update(self):
        x_traj_k = self.barrier_SGD_non_block()
            
        print("optimaizer")
        return x_traj_k

    def initial(self):
        self.beta = self.eta0
        if self.random_init:
            self.x0 = self.get_random_initial_point()
        else:
            self.x0 = self.x00
        f_0 = self.f(self.x0)


        
        time_0 = time() 
        (x_obstacle_trajectory, x_long_trajectory, constraints_long_trajectory, 
                            T_total, 
                            x_last) = self.log_barrier_decaying_eta()
        self.runtimes = [time() - time_0]

        if self.random_init:
                self.x0 = self.get_random_initial_point()
                f_0 = self.f(self.x0)
        else:
            self.x0 = self.x00

        x_obstacle_trajectory = self.obstacle
        
        x_long_trajectory = self.x0
        
        constraints_long_trajectory = np.max(self.h(self.x0))    
        T_total = 0
        
        self.eta = self.eta0
        x0 = self.x0
        x_prev = x0
        # print(self.f(self.x0))
          
        print("x_long_trajectory in log_barrier_decay", x0)
       
        print("constraints_long_trajectory in log_barrier_decay", constraints_long_trajectory)
        print("T_total in log_barrier_decay", T_total)


    def barrier_SGD_non_block(self):
        print("non block barrier_SGD")
        """
        Runs LB_SGD with constant parameter eta
        """
        if self.t_count == 0:
            self.xs = []
            # x_trajectory = [] # added
            xt = self.x0
            self.x_last = self.x0
            #print("im,xt",xt)
            # Tk = 0    
        
            
        #print("ORacle Xo", xt)
        self.oracle.sample(self.x_last)  
        
        self.step = self.dB_estimator()
        step_norm = np.linalg.norm(self.step)
        #print("step_norm", step_norm)
        gamma = self.compute_gamma(self.t_count)
        
        if not(step_norm < self.eta and self.no_break == False):
            

            xt = self.x_last - gamma * self.step # calculate and update policy
            #print("xt", xt)
            # Tk += 1
            
            x_trajectory = np.array([xt]) # # is the policy
            gamma_trajectory = np.array([gamma])
            
            constraints_trajectory = np.max(self.h(xt))
            worst_constraint = np.max(self.h(xt))
           
            self.t_count += 1
            if self.t_count == self.T:
                self.t_count = 0
                # print("x_trajectory", x_trajectory)
                #print("gammea_trajectory", gamma_trajectory)
                
                #print("constraints_trajectory", constraints_trajectory)
                #print("worst_constraint", worst_constraint)    

            self.xs.append(xt)
            self.x_last = xt
        if self.t_count == 0:
            self.x0 = self.x_last
            self.eta = self.eta * self.factor
        return xt#x_trajectory, gamma_trajectory,  constraints_trajectory, x_last, Tk
          
  


# @dataclass
class FhFunction:
    obstacle: np.array = [(50., 50., 10.)]
    obs_x_pos: float = 50.0
    obs_y_pos: float = 50.0
    diagonal_cost: float = 0.5
    k1: float = 5 # Gain for goal attraction
    k2: float = 300  # Gain for obstacle repulsion
    robot_goal: np.array = ([80., 80.])
    def move_obstacle(self,x,y):
        self.obs_x_pos += x
        self.obs_y_pos += y
        self.obstacle = [(self.obs_x_pos,self.obs_y_pos,10.0)]
    
    def setNewGoal(self, x, y):
        self.robot_goal = ([x,y])
    
    def h(self,x):
        repulsive_force = np.array([0., 0.])
        
        for obs in self.obstacle:
            diff = x - obs[:2]
            distance = np.linalg.norm(diff)
            
            if distance <= 1.5 * obs[2]:
                repulsive_force -=   diff - np.array([2,2])
                #print("repulsive_force",repulsive_force)
            else:
                repulsive_force = np.array([0., 0.])

                
        return repulsive_force

    def f(self,x):
        # Calculate the costs for each of the possible steps
        step_costs = [
            (x[0], x[1] + 1, 1),
            (x[0], x[1] - 1, 1),
            (x[0] + 1, x[1], 1),
            (x[0] - 1, x[1], 1),
            (x[0] + 1, x[1] + 1, self.diagonal_cost),
            (x[0] + 1, x[1] - 1, self.diagonal_cost),
            (x[0] - 1, x[1] + 1, self.diagonal_cost),
            (x[0] - 1, x[1] - 1, self.diagonal_cost)
        ]
        

        # Calculate the distance to the robot goal for each step and find the closest one
        closest_step = min(step_costs, key=lambda step: ((step[0] - self.robot_goal[0])**2 + (step[1] - self.robot_goal[1])**2)**0.5)

        repulsive_force = np.array([0., 0.])
        
        for obs in self.obstacle:
            diff = x - obs[:2]
            distance = np.linalg.norm(diff)
            
            
            if distance <= 2.4 * obs[2]:
                repulsive_force += self.k2 * ((2.4 * obs[2] - distance) / (distance**3)) * diff
    

            elif distance >= 2 * obs[2]:
                repulsive_force = np.array([0., 0.])   

        #print("repulsive_force",repulsive_force)     

        
        return  np.array([np.linalg.norm(closest_step[0] - self.robot_goal[0] + repulsive_force[0]), np.linalg.norm(closest_step[1] - self.robot_goal[1] + repulsive_force[1])]) 

# @dataclass
class Simulation:
    d: float = 2
    m: float = 2
    x00: np.array = np.array([5, 5])
    x0: np.array = None
    M0: float = 0.5 / d
    Ms: np.array = 0.0 * np.ones(m)
    sigma: float = 0.001
    hat_sigma: float = 0.01
    init_std: float = 0.5 
    eta0: float = 0.5
    eta: float = None
    step: np.array = None
    reg: float = None
    # x_opt: float = None
    T: int = 3
    K: int = None
    experiments_num: int = 2
    mu: float = None
    xs: list = None
    s: int = None
    convex: bool = None
    random_init: bool = False
    no_break: bool = True
    x_total: list = None
    errors_total: list = None
    constraints_total: list = None
    beta: float = None
    delta: float = 0.01
    factor: float = 0.5
    runtimes: list = None
    obstacle: np.array = ([80.0, 80.0, 10])
    n: int = 5
    n_iters: int = 800
    nu: float = 0.1
    grid_size = (100, 100)
    robot_start = ([5., 5.])
    robot_goal = ([80., 80.])
    robot_radius: float = 5
    fig, ax = plt.subplots()

    sim_time: float = 400.0
    step_time: float = 0.1

    def __init__(self):
        print("init simulation")
        self.myFhFunctions = FhFunction()
        self.my_oracle = Oracle(
            f = self.myFhFunctions.f,
            h = self.myFhFunctions.h, 
            sigma = self.sigma,
            hat_sigma = 0.01,
            delta = self.delta,
            m = self.m,
            d = self.d,
            nu = self.nu,
            zeroth_order = True,
            n = self.n)
        
        self.opt = SafeLogBarrierOptimizer(
            x00 = self.x00,
            x0 = self.x00,
            M0 = self.M0,
            Ms = self.Ms,
            sigma = self.my_oracle.sigma,
            hat_sigma = self.my_oracle.hat_sigma,
            init_std = self.init_std,
            eta0 = self.eta0,
            oracle = self.my_oracle,
            f = self.myFhFunctions.f,
            h = self.myFhFunctions.h,
            d = self.d,
            m = self.m,
            reg = 0.1,
            factor = self.factor,
            T = self.T,
            K = int(self.n_iters / self.T / 2. / self.n),
            experiments_num = self.experiments_num,
            mu = 0.,
            convex = True,
            random_init = True,
            no_break = False,
            obstacle = self.obstacle,
            )
        self.opt.initial()
        # self.opt.run_average_experiment()
        for x in range(1500):
            xt = self.opt.update()
            self.plot_nonblock(xt,self.myFhFunctions.obstacle)
            if self.reachedGoal(xt):
                x = np.random.random_integers(3,80)
                y = np.random.random_integers(3,80)
                self.myFhFunctions.setNewGoal(x,y)
                self.robot_goal = ([x, y])
            if x < 100:
                self.myFhFunctions.move_obstacle(-0.1,0.01)
            else:
                self.myFhFunctions.move_obstacle(0.1,0.01)
            if x == 200:
                self.myFhFunctions.setNewGoal(5.0,5.0)
                self.robot_goal = ([5.0, 5.0])
        # robot_pos = self.playback_run_simulation(self.sim_time, self.step_time, self.robot_start, self.robot_goal, self.robot_radius, self.obstacle, self.opt, self.update_plot)# Display the result
        plt.show()
        print("done")
    def reachedGoal(self,pos):
        goal = np.asarray(self.robot_goal)
        pos = np.asarray(pos)
        distance=math.sqrt(pow(goal[0]-pos[0],2)+pow(goal[1]-pos[1],2))
        if distance < 0.5:
            return True
        else:
            return False
    def update(self):
        print("Update Simulatio")
    def update_plot(self, ax, robot_pos, obstacle, robot_radius):
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])

        # Plot the obstacle(s)  
        for obs in obstacle:
            obs_circle = patches.Circle((obs[0], obs[1]), obs[2], color='red')
            ax.add_patch(obs_circle)

        # Plot the start, goal, and robot path
        ax.plot(*self.robot_start, 'go', markersize=10, label='Start')
        ax.plot(*self.robot_goal, 'bo', markersize=10, label='Goal')
        ax.plot(robot_pos[0], robot_pos[1], 'ro', markersize=robot_radius, label='Robot')
        line, = ax.plot([], [], 'b-', label='Robot Path')
        ax.legend()

    def plot_nonblock(self,robot_pos,obstacle):
        # total_force = self.opt.x_total[-1][-1]
        #print("exp_n", j,  "total_force", total_force)
        # Update the robot's position using the total force
        # robot_pos = total_force
        # self.update_plot(self.ax, robot_pos, [(self.opt.obstacle_list[-1][-1][0],self.opt.obstacle_list[-1][-1][1], self.opt.obstacle_list[-1][-1][2])], self.robot_radius)
        self.update_plot(self.ax, robot_pos, obstacle, self.robot_radius)
        plt.pause(0.001)


    def playback_run_simulation(self,sim_time, step_time, robot_pos, robot_goal, robot_radius, obstacle, optimizer, update_plot):
        #current_time = 0
        #while current_time < sim_time:
        # print("j length: ", len(plplp.x_total))
        print("o[0]",len(self.opt.obstacle_list[0]))
        print("ol[0]",len(self.opt.obstacle_list))
        print("asd0",self.opt.obstacle_list)
        print("obs", obstacle)
        count = 0
        print("obs2: ",self.opt.obstacle_list[count][count:count+3])
        for j in range(len(self.opt.x_total)):
            # print("j", j)
            #print("j", plplp.x_total[j]) 
            if j in [0,49,90]:
                for k in range(len(self.opt.x_total[j])):
                    
                    total_force = self.opt.x_total[j][k]
                    #print("exp_n", j,  "total_force", total_force)
                    # Update the robot's position using the total force
                    robot_pos = total_force
                    self.update_plot(self.ax, robot_pos, [(self.opt.obstacle_list[j][k][0],self.opt.obstacle_list[j][k][1], self.opt.obstacle_list[j][k][2])], robot_radius)
                    plt.pause(0.001)
                

            
            #current_time += step_time
        # print("j length: ",len(plplp.x_total[0]))
        # print("Xt array: ", len(plplp.x_total))
        # print("Xt :", plplp.x_total[0][0])
        
        np.save("runs.npy",self.opt.x_total) # what experiement we want to save
        print("before saved: ", self.opt.x_total[0][-1])
        pre_xt = np.load("runs.npy")
        print("saved", pre_xt[0][-1])
            
        return robot_pos