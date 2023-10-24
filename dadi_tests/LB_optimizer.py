# Write the documentation

import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp, conelp, coneqp
from scipy.stats import norm, chi2

import matplotlib.pyplot as plt
import matplotlib.lines as line
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
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
            self.constraints_grad = self.dh + np.random.normal(0, self.hat_sigma / self.n**0.5, (self.m, self.d))
            self.alphas = - self.constraints_values - \
                (np.log(1. / self.delta))**0.5 * self.sigma / self.n**0.5 / 2. * np.ones(self.m)


@dataclass
class SafeLogBarrierOptimizer:
    """
    This class allows to run LB-SGD optimization procedure given the oracle for the objective f and constraint h. 
    """
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
    x_opt: float = None
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
        gamma = min(1. / step_norm * np.min(alphas / ( 2 * L_dirs +  alphas_reg**0.5 * self.Ms**0.5)), 
                    1. / M2 )
        return gamma

    def dB_estimator(self):
        """
        Computes the log barrier gradient estimator
        """
        alphas = self.oracle.alphas
        jacobian = self.oracle.constraints_grad
        df_e = self.oracle.objective_grad
        denominators = 1. / np.maximum(np.ones(self.m) * self.reg, alphas)
        dB = df_e + self.eta * jacobian.T.dot(denominators)
        return dB
    
    def barrier_SGD(self):
        """
        Runs LB_SGD with constant parameter eta
        """
        self.xs = []
        xt = self.x0
        Tk = 0    
        for t in range(self.T):
            self.oracle.sample(xt)  
            self.step = self.dB_estimator()
            step_norm = np.linalg.norm(self.step)
            gamma = self.compute_gamma(t)

            if step_norm < self.eta and self.no_break == False:
                break

            xt = xt - gamma * self.step
            Tk += 1
            if t == 0:
                x_trajectory = np.array([xt])
                gamma_trajectory = np.array([gamma])
                errors_trajectory = self.f(xt) - self.f(self.x_opt)
                constraints_trajectory = np.max(self.h(xt))
                worst_constraint = np.max(self.h(xt))
            else:
                x_trajectory = np.vstack((x_trajectory, xt))
                gamma_trajectory = np.vstack((gamma_trajectory, gamma))
                errors_trajectory = np.hstack((errors_trajectory, self.f(xt) - self.f(self.x_opt)))
                constraints_trajectory = np.hstack((constraints_trajectory, np.max(self.h(xt))))
                worst_constraint = max(worst_constraint, np.max(self.h(xt)))

            self.xs.append(xt)
            x_last = xt
    
        return x_trajectory, gamma_trajectory, errors_trajectory, constraints_trajectory, x_last, Tk
          
    def log_barrier_decaying_eta(self):
        """
        Outer loop of LB-SGD with decreasing eta
        """
        f_opt = self.f(self.x_opt)
        x_long_trajectory = self.x0
        errors_long_trajectory = self.f(self.x0) - f_opt
        constraints_long_trajectory = np.max(self.h(self.x0))    
        T_total = 0
        
        self.eta = self.eta0
        x0 = self.x0
        x_prev = x0
        
        for k in range(self.K):
                
            x_traj_k, gamma_traj_k, errors_traj_k, constraints_traj_k, x_last_k, T_k = self.barrier_SGD()
            errors_long_trajectory = np.hstack((errors_long_trajectory, errors_traj_k))
            constraints_long_trajectory = np.hstack((constraints_long_trajectory, constraints_traj_k))
            x_long_trajectory = np.vstack((x_long_trajectory, x_traj_k))
            T_total = T_total + T_k
            self.x0 = x_last_k
            self.eta = self.eta * self.factor

        return x_long_trajectory, errors_long_trajectory, constraints_long_trajectory, T_total, x_last_k

    def get_random_initial_point(self):
        """
        Obtains random safe initial point
        """
        x0_det = self.x00
        for i in range(1000 * self.d):
            x0 =  x0_det + np.random.uniform(low=-1, high=1, size=self.d) * self.init_std
            if (self.h(x0) < - self.beta).all():
                break
        return x0
    
    def run_average_experiment(self):
        """
        Runs the LB_SGD multiple times, 
        
        Outputs: x_last, 
        Updates: errors_total, constraints_total, xs
        """
        self.beta = self.eta0
        if self.random_init:
            self.x0 = self.get_random_initial_point()
        else:
            self.x0 = self.x00

        f_0 = self.f(self.x0)
        f_opt = self.f(self.x_opt)
        
        time_0 = time() 
        (x_long_trajectory, errors_long_trajectory, 
                            constraints_long_trajectory, 
                            T_total, 
                            x_last) = self.log_barrier_decaying_eta()
        self.runtimes = [time() - time_0]

        x_total = []
        errors_total = []
        constraints_total = []

        errors_total.append(errors_long_trajectory)
        constraints_total.append(constraints_long_trajectory)
        
        for i in range(self.experiments_num - 1):
            if self.random_init:
                self.x0 = self.get_random_initial_point()
                f_0 = self.f(self.x0)
            else:
                self.x0 = self.x00
                
            time_0 = time() 
            (x_long_trajectory, errors_long_trajectory, 
                                constraints_long_trajectory, 
                                T_total, 
                                x_last) = self.log_barrier_decaying_eta()
            self.runtimes.append(time() - time_0)
            x_total.append(x_long_trajectory)
            errors_total.append(errors_long_trajectory)
            constraints_total.append(constraints_long_trajectory)
        self.x_total = x_total
        self.errors_total = errors_total
        self.constraints_total = constraints_total
        print('LB_SGD runs finished')
        return x_last
