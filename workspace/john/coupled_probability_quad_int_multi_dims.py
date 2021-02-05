# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:38:24 2021

@author: jkcle
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
path = r'C:\Users\jkcle\Documents\GitHub\Nonlinear-Statistical-Coupling\nsc\util'
sys.path.insert(1, path)
import function as f

import numpy as np
from typing import Any, List  # for NDArray types
from scipy.integrate import quad


def coupled_probability(density_func,
                        kappa: float = 0.0, 
                        alpha: float = 1.0, 
                        dim: int = 1,
                        support: tuple = (-np.inf, np.inf)) -> [float, Any]:

    
    # Calculate the risk-bias.
    kMult = (-alpha * kappa) / (1 + dim*kappa)
    
    def raised_density_func(x):
        return density_func(x) ** (1-kMult)
    
    # Calculate the normalization factor to the coupled CDF equals 1.
    division_factor = quad(raised_density_func, a=support[0], b=support[1])[0]
    
    
    # Define a function to calculate coupled densities
    def coupled_prob(values):
        return raised_density_func(values) / division_factor
    
    # Return the new functions that calculates the coupled density of a value.
    return coupled_prob


def coupled_cross_entropy(density_func_p, 
                          density_func_q, 
                          kappa: float = 0.0, 
                          alpha: float = 1.0, 
                          dim: int = 1,
                          support: tuple = (-np.inf, np.inf), 
                          root: bool = False) -> [float, Any]:
    
    # Fit a coupled_probability function to density_func_p with the other
    # given parameters.
    my_coupled_probability = coupled_probability(density_func=density_func_p,
                                                 kappa=kappa, 
                                                 alpha=alpha,
                                                 dim=dim, 
                                                 support=support)
    
    def raised_density_func_q(x):
        return density_func_q(x)**(-alpha)
    
    if root == False:
        
        def no_root_coupled_cross_entropy(x):
            return (my_coupled_probability(x)
                    *(1/-alpha)
                    *f.coupled_logarithm(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim))
        
        # Integrate the function.
        final_integration = -quad(no_root_coupled_cross_entropy, 
                                      a=support[0], 
                                      b=support[1])[0]
        
    else:
        def root_coupled_cross_entropy(x):
            return (my_coupled_probability(x)
                    *f.coupled_logarithm(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim)**(1/alpha))
        
        # Integrate the function.
        final_integration = quad(root_coupled_cross_entropy, 
                                     a=support[0], 
                                     b=support[1])[0]
        
    return final_integration


def coupled_entropy(density_func, 
                    kappa: float = 0.0, 
                    alpha: float = 1.0, 
                    dim: int = 1, 
                    support: tuple = (-np.inf, np.inf),
                    root: bool = False) -> [float, Any]:

    
    return coupled_cross_entropy(density_func, 
                                 density_func, 
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 dim=dim,
                                 support=support, 
                                 root=root)


def coupled_divergence(density_func_p, 
                       density_func_q, 
                       kappa: float = 0.0, 
                       alpha: float = 1.0, 
                       dim: int = 1, 
                       support: tuple = (-np.inf, np.inf),
                       root: bool = False) -> [float, Any]:

    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy(density_func_p,
                                                           density_func_q,
                                                           kappa=kappa,
                                                           alpha=alpha, 
                                                           dim=dim,
                                                           support=support, 
                                                           root=root)
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy(density_func_p, 
                                                kappa=kappa, 
                                                alpha=alpha, 
                                                dim=dim,
                                                support=support,
                                                root=root)
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p


def tsallis_entropy(density_func, 
                    kappa,
                    alpha = 1, 
                    dim = 1, 
                    support: tuple = (-np.inf, np.inf), 
                    normalize = False, 
                    root = False):

    
    if normalize:
        entropy = (1+kappa)**(1/alpha) * coupled_entropy(density_func,  
                                                         kappa=kappa, 
                                                         alpha=alpha, 
                                                         dim=dim, 
                                                         support=support,
                                                         root=root)
    else:
        def un_normalized_density_func(x):
            return density_func(x)**(1+(alpha*kappa/(1+kappa)))
        
        entropy = (quad(un_normalized_density_func, 
                            a=support[0], 
                            b=support[1])[0]
                       * (1+kappa)**(1/alpha)
                       * coupled_entropy(density_func,
                                         kappa=kappa,
                                         alpha=alpha,
                                         dim=dim,
                                         support=support,
                                         root=root))
    
    return entropy

def shannon_entropy(density_func, 
                    dim = 1, 
                    support: tuple = (-np.inf, np.inf),
                    root = False):
    
    if root:
        alpha = 2
    else: 
        alpha = 1
    
    return coupled_entropy(density_func,
                           kappa=0.0, 
                           alpha=alpha, 
                           dim=dim, 
                           support=support,
                           root=root)