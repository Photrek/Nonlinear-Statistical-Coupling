# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:38:24 2021

@author: jkcle
"""
import function_john as fj
import numpy as np
from typing import Any, List  # for NDArray types
from scipy.integrate import quad
from mpmath import nsum

def coupled_probabilityV1(density_func,
                          realized_support,
                          kappa: float = 0.0, 
                          alpha: float = 1.0, 
                          dim: int = 1,
                          support: tuple = (-np.inf, np.inf),
                          continuous: bool = True) -> [float, Any]:

    
    # Calculate the risk-bias.
    kMult = (-alpha * kappa) / (1 + dim*kappa)
    
    def raised_density_func(x):
        return density_func(x) ** (1-kMult)
    
    if continuous:
        # Calculate the normalization factor to the coupled CDF equals 1.
        division_factor = quad(raised_density_func, a=support[0], b=support[1])[0]
    
    else:
        # Calculate the normalization factor to the coupled CDF equals 1.
        division_factor = np.float64(nsum(raised_density_func, support))
        
    # Calculate the coupled densities
    coupled_dist = raised_density_func(realized_support) / division_factor

    return coupled_dist


def coupled_probability(density_func,
                        kappa: float = 0.0, 
                        alpha: float = 1.0, 
                        dim: int = 1,
                        support: tuple = (-np.inf, np.inf),
                        continuous: bool = True) -> [float, Any]:

    
    # Calculate the risk-bias.
    kMult = (-alpha * kappa) / (1 + dim*kappa)
    
    def raised_density_func(x):
        return density_func(x) ** (1-kMult)
    
    # Integrate the raised PDF for continuous random variables.
    if continuous:
        # Calculate the normalization factor to the coupled CDF equals 1.
        division_factor = quad(raised_density_func, a=support[0], b=support[1])[0]
    
    # Sum the raised PMF for discrete random variables.
    else:
        # Calculate the normalization factor to the coupled CDF equals 1.
        division_factor = np.float64(nsum(raised_density_func, support))
    
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
                          root: bool = False,
                          continuous: bool = True) -> [float, Any]:
    
    # Fit a coupled_probability function to density_func_p with the other
    # given parameters.
    my_coupled_probability = coupled_probability(density_func=density_func_p,
                                                 kappa=kappa, 
                                                 alpha=alpha,
                                                 dim=dim, 
                                                 support=support,
                                                 continuous=continuous)
    
    def raised_density_func_q(x):
        return density_func_q(x)**(-alpha)
    
    if root == False:
        
        def no_root_coupled_cross_entropy(x):
            return (my_coupled_probability(x)
                    *(1/-alpha)
                    *fj.coupled_logarithm(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim))
        
        # Integrate the function for continuous random variables.
        if continuous:
            # Integrate the function.
            final_integration = -quad(no_root_coupled_cross_entropy, 
                                      a=support[0], 
                                      b=support[1])[0]
            
        # Sum the function for discrete random variables.
        else:
            # Sum the function. The variable is still labeled as 
            # final_integration, but it is a sum.
            final_integration = -np.float64(nsum(no_root_coupled_cross_entropy, 
                                                 support))
        
    else:
        def root_coupled_cross_entropy(x):
            return (my_coupled_probability(x)
                    *fj.coupled_logarithm(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim)**(1/alpha))
        
        # Integrate the function for continuous random variables.
        if continuous:
            # Integrate the function.
            final_integration = quad(root_coupled_cross_entropy, 
                                     a=support[0], 
                                     b=support[1])[0]
            
        # Sum the function for discrete random variables.
        else:
            # Sum the function. The variable is still labeled as 
            # final_integration, but it is a sum.
            final_integration = np.float64(nsum(no_root_coupled_cross_entropy, 
                                                support))
        
    return final_integration


def coupled_entropy(density_func, 
                    kappa: float = 0.0, 
                    alpha: float = 1.0, 
                    dim: int = 1, 
                    support: tuple = (-np.inf, np.inf),
                    root: bool = False,
                    continuous: bool = True) -> [float, Any]:

    
    return coupled_cross_entropy(density_func, 
                                 density_func, 
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 dim=dim,
                                 support=support, 
                                 root=root,
                                 continuous=continuous)


def coupled_divergence(density_func_p, 
                       density_func_q, 
                       kappa: float = 0.0, 
                       alpha: float = 1.0, 
                       dim: int = 1, 
                       support: tuple = (-np.inf, np.inf),
                       root: bool = False,
                       continuous: bool = True) -> [float, Any]:

    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy(density_func_p,
                                                           density_func_q,
                                                           kappa=kappa,
                                                           alpha=alpha, 
                                                           dim=dim,
                                                           support=support, 
                                                           root=root,
                                                           continuous=continuous)
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy(density_func_p, 
                                                kappa=kappa, 
                                                alpha=alpha, 
                                                dim=dim,
                                                support=support,
                                                root=root,
                                                continuous=continuous)
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p


def tsallis_entropy(density_func, 
                    kappa,
                    alpha = 1, 
                    dim = 1, 
                    support: tuple = (-np.inf, np.inf), 
                    normalize = False, 
                    root = False,
                    continuous: bool = True):

    
    if normalize:
        entropy = (1+kappa)**(1/alpha) * coupled_entropy(density_func,  
                                                         kappa=kappa, 
                                                         alpha=alpha, 
                                                         dim=dim, 
                                                         support=support,
                                                         root=root,
                                                         continuous=continuous)
    else:
        def un_normalized_density_func(x):
            return density_func(x)**(1+(alpha*kappa/(1+kappa)))
        
        if continuous:
            entropy = (quad(un_normalized_density_func, 
                            a=support[0], 
                            b=support[1])[0]
                       * (1+kappa)**(1/alpha)
                       * coupled_entropy(density_func,
                                         kappa=kappa,
                                         alpha=alpha,
                                         dim=dim,
                                         support=support,
                                         root=root,
                                         continuous=continuous))
        
        else:
            entropy = (np.float64(nsum(un_normalized_density_func, support))
                       * (1+kappa)**(1/alpha)
                       * coupled_entropy(density_func,
                                         kappa=kappa,
                                         alpha=alpha,
                                         dim=dim,
                                         support=support,
                                         root=root,
                                         continuous=continuous))
    
    return entropy

def shannon_entropy(density_func, 
                    dim = 1, 
                    support: tuple = (-np.inf, np.inf),
                    root = False,
                    continuous: bool = True):
    
    if root:
        alpha = 2
    else: 
        alpha = 1
    
    return coupled_entropy(density_func,
                           kappa=0.0, 
                           alpha=alpha, 
                           dim=dim, 
                           support=support,
                           root=root,
                           continuous=continuous)