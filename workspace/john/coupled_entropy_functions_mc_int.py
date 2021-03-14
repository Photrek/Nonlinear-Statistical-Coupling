# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:38:24 2021

@author: jkcle
"""

import nsc
import numpy as np
from typing import Any, List  # for NDArray types
from scipy.integrate import nquad


def importance_sampling_integrator(function, pdf, sampler, n=10000):
    """
    

    Parameters
    ----------
    function : TYPE
        DESCRIPTION.
    pdf : TYPE
        DESCRIPTION.
    sampler : TYPE
        DESCRIPTION.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Generate n samples from the probability distribution.
    samples = sampler(n)
    # Evaluate the function at the samples and divide by the probability 
    # density of the distribution at those samples.
    sampled_values = function(samples) / pdf(samples)
    # Return the mean of the samples as the MC estimate of the integral.
    return np.mean(sampled_values)


def coupled_probability(density_func,
                        sampler,
                        kappa = 0.0, 
                        alpha = 1.0, 
                        dim = 1,
                        n = 10000):
    """
    

    Parameters
    ----------
    density_func : TYPE
        DESCRIPTION.
    sampler : TYPE
        DESCRIPTION.
    kappa : TYPE, optional
        DESCRIPTION. The default is 0.0.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.0.
    dim : TYPE, optional
        DESCRIPTION. The default is 1.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    
    # Calculate the risk-bias.
    kMult = (-alpha * kappa) / (1 + dim*kappa)
    
    def raised_density_func(x):
        return density_func(x) ** (1-kMult)
    
    def raised_density_func_integration(*args):
        if dim == 1:
            x = np.array(args)
        else:
            x = np.array([args]).reshape(1, dim)
        return density_func(x) ** (1-kMult)
    
    # Calculate the normalization factor to the coupled CDF equals 1.
    division_factor = importance_sampling_integrator(raised_density_func_integration, 
                                                     pdf=density_func,
                                                     sampler=sampler, 
                                                     n=n)
    
    
    # Define a function to calculate coupled densities
    def coupled_prob(values):
        return raised_density_func(values) / division_factor
    
    # Return the new functions that calculates the coupled density of a value.
    return coupled_prob


def coupled_cross_entropy(density_func_p, 
                          density_func_q, 
                          sampler_p,
                          kappa: float = 0.0, 
                          alpha: float = 1.0, 
                          dim: int = 1,
                          root: bool = False,
                          n=10000) -> [float, Any]:
    
    # Fit a coupled_probability function to density_func_p with the other
    # given parameters.
    my_coupled_probability = coupled_probability(density_func=density_func_p,
                                                 sampler=sampler_p,
                                                 kappa=kappa, 
                                                 alpha=alpha,
                                                 dim=dim, 
                                                 n=n)
    
    def raised_density_func_q(x):
        return density_func_q(x)**(-alpha)
    
    if root == False:
        
        def no_root_coupled_cross_entropy(*args):
            if dim == 1:
                x = np.array(args)
            else:
                x = np.array([args]).reshape(1, dim)
            
            return (my_coupled_probability(x)
                    *(1/-alpha)
                    *nsc.log(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim))
        
        # Integrate the function.
        final_integration = -importance_sampling_integrator(no_root_coupled_cross_entropy, 
                                                            pdf=density_func_p,
                                                            sampler=sampler_p, 
                                                            n=n)
        
    else:
        def root_coupled_cross_entropy(*args):
            if dim == 1:
                x = np.array(args)
            else:
                x = np.array([args]).reshape(1, dim)
            return (my_coupled_probability(x)
                    *nsc.log(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim)**(1/alpha))
        
        # Integrate the function.
        final_integration = importance_sampling_integrator(root_coupled_cross_entropy, 
                                                           pdf=density_func_p,
                                                           sampler=sampler_p, 
                                                           n=n)
        
    return final_integration


def coupled_entropy(density_func, 
                    sampler,
                    kappa: float = 0.0, 
                    alpha: float = 1.0, 
                    dim: int = 1, 
                    root: bool = False,
                    n=10000) -> [float, Any]:

    
    return coupled_cross_entropy(density_func, 
                                 density_func, 
                                 sampler_p=sampler,
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 dim=dim,
                                 root=root,
                                 n=n)


def coupled_divergence(density_func_p, 
                       density_func_q, 
                       sampler_p,
                       kappa: float = 0.0, 
                       alpha: float = 1.0, 
                       dim: int = 1, 
                       root: bool = False,
                       n=10000) -> [float, Any]:

    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy(density_func_p,
                                                           density_func_q,
                                                           sampler_p=sampler_p,
                                                           kappa=kappa,
                                                           alpha=alpha, 
                                                           dim=dim,
                                                           root=root,
                                                           n=n)
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy(density_func_p, 
                                                sampler=sampler_p,
                                                kappa=kappa, 
                                                alpha=alpha, 
                                                dim=dim,
                                                root=root,
                                                n=n)
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p


def tsallis_entropy(density_func, 
                    sampler,
                    kappa,
                    alpha = 1, 
                    dim: int = 1, 
                    normalize = False, 
                    root = False,
                    n=10000):

    
    if normalize:
        entropy = (1+kappa)**(1/alpha) * coupled_entropy(density_func,  
                                                         sampler,
                                                         kappa=kappa, 
                                                         alpha=alpha, 
                                                         dim=dim, 
                                                         root=root,
                                                         n=n)
    else:
        def un_normalized_density_func(*args):
            if dim == 1:
                x = np.array(args)
            else:
                x = np.array([args]).reshape(1, dim)
            return density_func(x)**(1+(alpha*kappa/(1+kappa)))
        
        entropy = (importance_sampling_integrator(un_normalized_density_func, 
                                                  pdf=density_func, 
                                                  sampler=sampler, 
                                                  n=n)
                       * (1+kappa)**(1/alpha)
                       * coupled_entropy(density_func,
                                         sampler=sampler,
                                         kappa=kappa,
                                         alpha=alpha,
                                         dim=dim,
                                         root=root,
                                         n=n))
    
    return entropy

def shannon_entropy(density_func, 
                    sampler,
                    dim: int = 1, 
                    root = False,
                    n=10000):
    
    if root:
        alpha = 2
    else: 
        alpha = 1
    
    return coupled_entropy(density_func,
                           sampler,
                           kappa=0.0, 
                           alpha=alpha, 
                           dim=dim, 
                           root=root,
                           n=10000)