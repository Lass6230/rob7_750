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
        # print("x sample1",x)
        self.objective_value = self.f(x) + np.random.normal(0, self.sigma / self.n**0.5)
        # print("x sample2",x)
        # print("obs value",self.objective_value)
        self.constraints_values = self.h(x) + np.random.normal(0, self.sigma / self.n**0.5, self.m)
        # print("x sample3",x)
        self.df = nd.Gradient((self.f)(x))
        # print("x sample4",x)
        self.dh = nd.Gradient((self.h)(x))
        # print("x sample5",x)
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
    # x_array_last = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
    
    sample_time: float = 0.1
    horizon: int = 40
    x_array_last = np.zeros((horizon,3))
    x_last = None
    u_last = [0.0,0.0,0.0]
    dbLast = np.array([0.01,0.01,0.01])
    cmd_vel = None
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
    g: Callable = None 
    predicted_horizon: Callable = None
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
    factor: float = None
    runtimes: list = None
    t_count: int = 0
    previous_time: float = 0.0
    jacobian_t_dot = None
    correction_factor = np.array([1.,1.,1.])
    
    
    def correctDirection(self):
         self.correction_factor = np.array([1.,1.,1.])
         if self.eta > 0.5:
            if self.jacobian_t_dot[0] > 0.:
                self.correction_factor[0] = 2./self.eta
            else:
                self.correction_factor[0] = 1.

            if self.jacobian_t_dot[1] > 0.:
                self.correction_factor[1] = 2./self.eta
            else:
                self.correction_factor[1] = 1.

            if self.jacobian_t_dot[2] > 0.:
                self.correction_factor[2] = 2/self.eta
            else:
                self.correction_factor[2] = 1.
             

         return self.correction_factor

    
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
        
        # print("gamma", gamma)
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
        self.jacobian_t_dot = jacobian.T.dot(denominators)
        
        self.correctDirection()

        # print("COREEECT", self.correction_factor)

        dB = df_e + self.correction_factor*self.eta * jacobian.T.dot(denominators)
        #dB = df_e + self.eta * jacobian.T.dot(denominators)
        

        """if dB[0]  < 0.0:elf.st
            dB[0] = self.dbLast[0]

        if dB[1]  < 0.0:
            dB[1] = self.dbLast[1]

        if dB[2]  < 0.0:
            dB[2] = self.dbLast[2]        elf.st
        self.dbLast = dB"""
        # print("jacobian", jacobian.T.dot(denominators))
        # print("DB", dB)
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
            elif x_trajectory != None:
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
          
        #print("x_long_trajectory in log_barrier_decay", x0)
       
        #print("constraints_long_trajectory in log_barrier_decay", constraints_long_trajectory)
        #print("T_total in log_barrier_decay", T_total)
        
        
        for k in range(self.K):
                
            x_traj_k, gamma_traj_k, constraints_traj_k, x_last_k, T_k = self.barrier_SGD()
            
            constraints_long_trajectory = np.hstack((constraints_long_trajectory, constraints_traj_k))
            x_long_trajectory = np.vstack((x_long_trajectory, x_traj_k))
            T_total = T_total + T_k
            self.x0 = x_last_k
            #print(self.h(self.x0))

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
        # print("online traning on previouss online traninged mode stuff :P")
        
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
        # print('LB_SGD runs finished')

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
        print(x_total)
        self.constraints_total = constraints_total
        # print('LB_SGD runs finished')

        return x_last
    

    
    def update(self):
        x_traj_k = self.barrier_SGD_non_block()
            
        # print("optimaizer")
        return x_traj_k

    def initial(self):
        # self.beta = self.eta0
        # if self.random_init:
        #     self.x0 = self.get_random_initial_point()
        # else:
        #     self.x0 = self.x00
        # f_0 = self.f(self.x0)


        
        # time_0 = time() 
        # (x_obstacle_trajectory, x_long_trajectory, constraints_long_trajectory, 
        #                     T_total, 
        #                     x_last) = self.log_barrier_decaying_eta()
        # self.runtimes = [time() - time_0]
        
        # if self.random_init:
        #         self.x0 = self.get_random_initial_point()
        #         f_0 = self.f(self.x0)
        # else:
        #     self.x0 = self.x00
        
        # x_obstacle_trajectory = self.obstacle
        
        # x_long_trajectory = self.x0
        
        # constraints_long_trajectory = np.max(self.h(self.x0))    
        # T_total = 0
        
        self.eta = self.eta0
        x0 = self.x0
        x_prev = x0
        # print(self.f(self.x0))
          
        # print("x_long_trajectory in log_barrier_decay", x0)
       
        # print("constraints_long_trajectory in log_barrier_decay", constraints_long_trajectory)
        # print("T_total in log_barrier_decay", T_total)


    def barrier_SGD_non_block(self):
        # print("non block barrier_SGD")
        """
        Runs LB_SGD with constant parameter eta
        """
        if self.t_count == 0:
            self.xs = []
            # x_trajectory = [] # added
            xt = self.x0
            self.x0 = self.x_last
            # self.x_last = self.x0

            #print("im,xt",xt)
            # Tk = 0    
        
            
        #print("ORacle Xo", xt)
        self.oracle.sample(self.x_array_last) # maybe change to self.u_last 
        
        self.step = self.dB_estimator()
        step_norm = np.linalg.norm(self.step)
        #print("step_norm", step_norm)
        gamma = self.compute_gamma(self.t_count)
        
        # if not(step_norm < self.eta and self.no_break == False):
        ut = []
        xt = []
        xt.append(self.x_last)
        # ut.append(self.u_last)
        for h in range(self.horizon):
            xt.append(xt[h]-gamma*self.step)
            ut.append((xt[h+1]-xt[h])/self.sample_time)
        
        # xt = self.x_last - gamma * self.step # calculate and update policy
        # self.u_last = ut[0]
        self.cmd_vel = ut[0]
        self.x_array_last = xt
        self.x_array_last.pop(0)
        # print("x_last_array",self.x_array_last)

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



        self.xs.append(xt[1])
        # self.x_last = xt
        """if self.t_count == 0:
            self.x0 = self.x_last
            if self.h(xt)[0] <= -0.02:
                self.factor = 0.8
            elif self.h(xt)[0] > -0.02 and self.h(xt)[0] <= -0.01:
                self.factor = 0.99
            elif self.h(xt)[0] > -0.01:        
                self.factor = 1.2


            self.eta = self.eta * self.factor
            if self.eta <= 0.00000000000001:
                self.eta = 0.00000000000001"""
        
        self.eta = (3/(1+np.exp(-5*(max(self.h(xt))*100+0.5)))*self.eta + 1.5/(1+np.exp(-2*(max(self.h(xt))*100+1)))*self.eta + 0.001)

        #self.eta = (0.5/(1+np.exp(-1*(max(self.h(xt))*100+0.5))))

        #self.eta = self.eta * self.factor

        if self.eta >= 1000:
            self.eta = 1000

            


        # print("is smakll plz", (max(self.h(xt))))
        #print("IS i work?",4/(1+np.exp(-5*(max(self.h(xt))*100+0.5)))*self.eta + 1/(1+np.exp(-1*(max(self.h(xt))*100+0.5)))*self.eta+0.0001)


        # print("WE ETA SPAGETT TONIGHT", self.eta)

        self.previous_time = time()
        return xt[1]#x_trajectory, gamma_trajectory,  constraints_trajectory, x_last, Tk
          
  


# @dataclass
class FhFunction:
    horizon: int = 40
    sample_time: float = 0.1
    obstacle: np.array = [(500., 500., 50.)]
    obs_x_pos: float = 500.
    obs_y_pos: float = 500.
    diagonal_cost: float = 0.5
    k1: float = 5 # Gain for goal attraction
    k2: float = 30 # Gain for obstacle repulsion                    
    robot_goal: np.array = ([800., 800., 0.0])
    theta: float = 0.0  # in radians
    wheel_radius: float = 1.0  # adjust as needed
    wheel_distance: float = 2.0  # adjust as needed
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    j: int = 0
    x_pos: float = 0.0
    y_pos: float = 0.0
    rot_pos: float = 0.0
    ok_distance: float = 0.3
    closest_points: np.array = None

    closest_points_field_1: np.array = None
    closest_points_field_2: np.array = None
    closest_points_field_3: np.array = None

    def __init__(self, ok_distance,horizon):
        self.ok_distance = ok_distance
        self.horizon = horizon

    def move_obstacle(self,x,y):
        self.obs_x_pos += x
        self.obs_y_pos += y
        self.obstacle = [(self.obs_x_pos,self.obs_y_pos,50.0)]
    
    def setNewGoal(self, x, y, rot):
        self.robot_goal = ([x,y,rot])
    
    def setPos(self, x, y, rot):
        self.x_pos = x
        self.y_pos = y
        self.rot_pos = rot

    def h(self, x):
        get_away = np.array([0.0000001,0.000001,0.00001])
        close_point_array = []

 
        closest_obstacle = self.closest_points
        cl_obs_1 = self.closest_points_field_1
        cl_obs_2 = self.closest_points_field_2
        cl_obs_3 = self.closest_points_field_3



        length_to_obs = []
        length_to_obs_1 = [0.0]
        length_to_obs_2 = [0.0]
        length_to_obs_3 = [0.0]

        if closest_obstacle is not None:   
            if len(closest_obstacle) != 0:
                length_to_obs_h = []
                for i in range(self.horizon):
                    length_to_obs_1 = [0.0]
                    for point in closest_obstacle:
                        length_to_obs_1.append(1 - np.linalg.norm(np.array(point) - np.array(x[i])[:2])) 
                    length_to_obs_h.append(0.01*(sum(length_to_obs_1)/len(length_to_obs_1)))
            # if len(cl_obs_1) != 0:
            #     for point in cl_obs_2:
            #         length_to_obs_2.append(1 - np.linalg.norm(np.array(point) - np.array(x[0])[:2]))
            # if len(cl_obs_1) != 0:
            #     for point in cl_obs_3:
            #         length_to_obs_3.append(1 - np.linalg.norm(np.array(point) - np.array(x[0])[:2]))
            # get_away =  np.array([, 0.01*(sum(length_to_obs_2)/len(length_to_obs_2)), 0.01*(sum(length_to_obs_3)/len(length_to_obs_3))])
        else:
            length_to_obs_h = []
            for i in range(self.horizon):
                length_to_obs_1 = [0.0]
                    
                length_to_obs_h.append(0.0)
        
        return length_to_obs_h
        #print("get awayyyy", get_away)
        # return np.array(get_away)


        
        
        # if closest_obstacle is not None:
        #     for point in closest_obstacle:
        #         length_to_obs = np.linalg.norm(np.array(point) - np.array(x)[:2])
        #         if length_to_obs <= 2:
        #             close_point_array.append(point)
        #         elif length_to_obs is not None and length_to_obs>2:
        #             close_point_array.clear()
        #             for p in close_point_array:
        #                 distances_x = [np.linalg.norm(p[0] - x[0]) ]
        #                 distances_y = [np.linalg.norm(p[1] - x[1])]
        #                 base = np.array(-distances_x,-distances_y, angle_diff)
        #                 print("I DO NOTHING", base)
        #                 return base
        #         else:
        #             return np.array([0.0,0.0,0.0])        

        #     if close_point_array:
                
        #         apple = min(p for p in close_point_array)
        #         # print("APPPLE TIME",apple)
        #         distances_x = [np.linalg.norm(apple[0] - x[0]) ]
        #         distances_y = [np.linalg.norm(apple[1] - x[1])]
        #         min_distance_x = min(distances_x)
        #         min_distance_y = min(distances_y)
        #         get_away = 0.003 * np.array([min_distance_x, min_distance_y, angle_diff])
        #         print("This smells", get_away)
        #         return get_away
            
        # return np.array([0.0,0.0,0.0]) 

        # step_costs = [self.linear_vel * math.cos(self.theta),
        #                     self.linear_vel * math.sin(self.theta),
        #                     self.angular_vel / self.wheel_distance]
                
        # target_angle = math.atan2(self.robot_goal[1] - x[1], self.robot_goal[0] - x[0])
        
        # angle_diff = target_angle - self.theta

        # while angle_diff > math.pi:
        #     angle_diff -= 2 * math.pi
        # while angle_diff < -math.pi:
        #     angle_diff += 2 * math.pi

        # # Set angular velocity proportional to the angle difference
        # self.angular_vel = 0.05 * angle_diff


        
        # # Set linear velocity proportional to the distance to the target
        # distance_to_target = math.sqrt((self.robot_goal[0] - x[0])**2 + (self.robot_goal[1] - x[1])**2)
        # self.linear_vel = 0.09 * distance_to_target    

        # # print("velocity", self.linear_vel)

        # #closest_step = min(step_costs, key=lambda step: (distance_to_target - step[0])**2 + (angle_diff - step[1])**2)
        # repulsive_force = np.array([0., 0.])
        # new_pos = np.array([0., 0.])

            

        # repulsive_force = np.array([0., 0.])
        # stop_force = np.array([0., 0.])
        # kill_force = np.array([0., 0.])

        # # for obs in self.obstacle:
        # #     diff = x - obs[:2]
        # #     distance = np.linalg.norm(diff)

        # #     if distance <= 8* obs[2]:
        # #         repulsive_force += 10000 * ((2.4 * obs[2] - distance)) * diff

        # #         return repulsive_force

        # #     if distance <= 5 * obs[2]:
        # #         self.j = 1
        # #     elif distance >= 8 * obs[2]:
        # #         self.j = 0    
            
            

        #     # while self.j == 1:
        #     #     # Calculate the vector from the robot to the obstacle
        #     #     diff_to_obs = x - obs[:2]

        #     #     # Calculate the tangent direction around the obstacle
        #     #     tangent = np.array([diff_to_obs[1]- 100 , -diff_to_obs[0]])

        #     #     # Normalize the tangent vector
        #     #     tangent /= np.linalg.norm(tangent)

        #     #     # Calculate the desired position around the obstacle using the tangent
        #     #     next_point_of_interest = x + 5 * tangent  # Adjust the distance as needed

        #     #     print("Next Point of Interest:", next_point_of_interest)

                


        #     #     # Return the distance to the next point of interest as a guidance parameter
        #     #     return next_point_of_interest * 0.9
        # return np.array([0.0,0.0,0.0])    

    def f(self, x):
        # print("x_array: ",x)

        lin_factor = 0.015
        lin_factor_y= 0.015
        ang_factor = 0.005  
        distance_to_target = np.array([0.0,0.0,0.0])

        for i in range(self.horizon):

            target_angle = math.atan2((self.robot_goal[1] - x[i][1]), (self.robot_goal[0] - x[i][0]))
        
       
            angle_diff = np.linalg.norm(target_angle - x[i][2])
        

            self.angular_vel = angle_diff
      


        
        
            distance_to_target += np.array([lin_factor*np.linalg.norm(self.robot_goal[0] - x[i][0]) , lin_factor_y*np.linalg.norm(self.robot_goal[1] - x[i][1]), ang_factor*self.angular_vel])
         
        self.linear_vel = distance_to_target 
  

        return distance_to_target
    
    def predicted_horizon(self, xt, ut):
        x = []
        x.append(xt)
        for i in range(self.horizon):
            x.append(self.g(x[i],ut))
        return x
    
    def g(self,xt,ut):
        g_t = [[np.cos(xt[2]),-np.sin(xt[2]),0],[np.sin(xt[2]),np.cos(xt[2]),0],[0,0,1]] # think dynamic matrix
        xt_1 = xt+self.sample_time*g_t*ut
        return xt_1
# @dataclass
class Simulation:
    d: float = 3
    m: float = 40 # needs to same value as horizon
    x00: np.array = np.array([0.0, 0.0])  ####### change this
    x0: np.array = None
    M0: float = 0.5 / d
    Ms: np.array = 0.1 * np.ones(m)
    sigma: float = 0.0000005
    hat_sigma: float = 0.00001
    init_std: float = 0.00005 
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
    delta: float = 0.05
    factor: float = 0.5
    runtimes: list = None
    obstacle: np.array = ([500.0, 500.0, 50])
    n: int = 5
    n_iters: int = 800
    nu: float = 0.1
    grid_size = (1000, 1000)
    robot_start = ([0., 0.])
    robot_goal = ([800., 800.])
    robot_radius: float = 5
    fig, ax = plt.subplots()
    theta: float = 0.  # in radians
    wheel_radius: float = 1.0  # adjust as needed
    wheel_distance: float = 2.0  # adjust as needed
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    ok_distance: float = 0.3

    sim_time: float = 400.0
    step_time: float = 0.1

    def __init__(self, ok_distance):
        self.ok_distance = ok_distance
        print("init simulation")
        self.myFhFunctions = FhFunction(
            ok_distance = self.ok_distance,
            horizon=self.m
        )
        self.my_oracle = Oracle(
            f = self.myFhFunctions.f,
            h = self.myFhFunctions.h, 
            sigma = self.sigma,
            hat_sigma = self.hat_sigma,
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
            horizon=self.m,
            sigma = self.my_oracle.sigma,
            hat_sigma = self.my_oracle.hat_sigma,
            init_std = self.init_std,
            eta0 = self.eta0,
            oracle = self.my_oracle,
            f = self.myFhFunctions.f,
            h = self.myFhFunctions.h,
            # g_t = self.myFhFunctions.g,
            predicted_horizon = self.myFhFunctions.predicted_horizon,
            d = self.d,
            m = self.m,
            reg = 0.1,
            factor = self.factor,
            T = self.T,
            K = int(self.n_iters / self.T / 2. / self.n),
            experiments_num = self.experiments_num,
            mu = 0.,
            convex = False,
            random_init = True,
            no_break = False,
            obstacle = self.obstacle,
            )
        self.myFhFunctions.setNewGoal(2.0,-2.0, 0.0)
        self.opt.initial()
        
        
        # # self.opt.run_average_experiment()
        # for x in range(1500):
        #     xt = self.opt.update()
        #     self.plot_nonblock(xt,self.myFhFunctions.obstacle)
        #     if self.reachedGoal(xt):
        #         x = np.random.random_integers(50,900)
        #         y = np.random.random_integers(50,900)
        #         self.myFhFunctions.setNewGoal(x,y)
        #         self.robot_goal = ([x, y])


        #     if x <= 1000:
        #         self.myFhFunctions.move_obstacle(0,0)
        #     """elif x <= 400 and x > 200:
        #         self.myFhFunctions.move_obstacle(1,1)
        #     elif x <= 600 and x > 400:
        #         self.myFhFunctions.factormove_obstacle(0,1)
        #     elif x <= 800 and x > 600:
        #         self.myFhFunctions.move_obstacle(1,0)
        #     elif x <= 1000 and x > 800:
        #         self.myFhFunctions.move_obstacle(0,-1) """    
        #     #else:
        #     #   self.myFhFunctions.move_obstacle(1,1)
        #     """if x == 700:
        #         self.myFhFunctions.setNewGoal(5.0,5.0)
        #         self.robot_goal = ([5.0, 5.0])"""
        # # robot_pos = self.playback_run_simulation(self.sim_time, self.step_time, self.robot_start, self.robot_goal, self.robot_radius, self.obstacle, self.opt, self.update_plot)# Display the result
        # plt.show()
        # print("done")
    
    def setGoal(self,x,y,rot):
        self.myFhFunctions.setNewGoal(x,y,rot)
        robot_goal = ([x, y, rot])
    # def setStart(self,x,y)

    def closest_arrays_to_zero(self,arrays, n,x,y,flattened_obs_zone_1 ,flattened_obs_zone_2, flattened_obs_zone_3):
        filtered_arrays = [item for item in arrays if isinstance(item, list) != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        
        filtered_zone_1 = [item for item in flattened_obs_zone_1 if isinstance(item, list) != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        filtered_zone_2 = [item for item in flattened_obs_zone_2 if isinstance(item, list) != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        filtered_zone_3 = [item for item in flattened_obs_zone_3 if isinstance(item, list) != [float('inf'), float('inf')] and item !=[float('-inf'), float('-inf')]and item != [float('-inf'), float('inf')]and item != [float('inf'), float('-inf')]]
        distances_1 = [(np.linalg.norm(np.array(array)-(x,y)), array) for array in filtered_zone_1]
        distances_1.sort(key=lambda x: x[0])  # Sort distances from smallest to largest
        distances_2 = [(np.linalg.norm(np.array(array)-(x,y)), array) for array in filtered_zone_2]
        distances_2.sort(key=lambda x: x[0])  # Sort distances from smallest to largest
        distances_3 = [(np.linalg.norm(np.array(array)-(x,y)), array) for array in filtered_zone_3]
        distances_3.sort(key=lambda x: x[0])  # Sort distances from smallest to largest
        

        
        distances = [(np.linalg.norm(np.array(array)-(x,y)), array) for array in filtered_arrays]
        distances.sort(key=lambda x: x[0])  # Sort distances from smallest to largest
        
        closest_n_arrays = [array for distance, array in distances[:n]]

        self.myFhFunctions.closest_points_field_1 = [array for distance, array in distances_1[:n]]
        self.myFhFunctions.closest_points_field_2 = [array for distance, array in distances_2[:n]]
        self.myFhFunctions.closest_points_field_3 = [array for distance, array in distances_3[:n]]
        self.myFhFunctions.closest_points = closest_n_arrays
        return closest_n_arrays , self.myFhFunctions.closest_points_field_1,self.myFhFunctions.closest_points_field_2,self.myFhFunctions.closest_points_field_3


    def setObstacles(self, obstacles):
        self.myFhFunctions.obstacle = obstacles

    def reachedGoal(self,pos, ok_distance):
        goal = np.asarray(self.robot_goal)
        pos = np.asarray(pos)
        distance=math.sqrt(pow(goal[0]-pos[0],2)+pow(goal[1]-pos[1],2))
        if distance < ok_distance:
            return True
        else:
            return False
    
    def setPos(self,x,y,rot):
        self.opt.x_last = [x,y,rot]
        
    
    def getCmdVel(self):
        return self.opt.cmd_vel[0], self.opt.cmd_vel[1], self.opt.cmd_vel[2]


    def update(self):
        xt = self.opt.update()
        return xt
    
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