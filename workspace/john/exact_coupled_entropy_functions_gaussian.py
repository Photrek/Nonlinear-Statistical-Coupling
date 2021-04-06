# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:11:20 2021

@author: jkcle
"""
import nsc
from nsc.distributions import CoupledNormal, MultivariateCoupledNormal
from numpy.linalg import det
import numpy as np
from math import gamma
from typing import Any, List  # for NDArray types

def importance_sampling_integrator(function, pdf, sampler, n=10000, rounds=1, seed=1):
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
    rounds : int
        DESCRIPTION. The default is 5.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Set a random seed.
    np.random.seed(seed)
    
    # Create a list to hold the estimates for each round.
    estimates = []
    
    for i in range(rounds):
        # Generate n samples from the probability distribution.
        samples = sampler(n)
        # Evaluate the function at the samples and divide by the probability 
        # density of the distribution at those samples.
        sampled_values = function(samples) / pdf(samples)
        # Add the estimate of the integral to the estimates list.
        estimates.append(np.mean(sampled_values))
    
    # Return the mean of the estimates as the estimate of the integral.
    return np.mean(estimates)

def coupled_normal_entropy(Sigma, kappa):
    """
    This function calculates the coupled entropy of a coupled Gaussian 
    distribution using its Sigma matrix and kappa value.
    Parameters
    ----------
    Sigma : numpy ndarray
        The equivalent of a covariance matrix for a coupled Gaussian 
        distribution.
    kappa : float
        A positive coupling value.
    Returns
    -------
    entropy : float
        The coupled entropy of the coupled Gaussian distribution with the 
        covariance matrix equivalent of Sigma and coupling value kappa.
    """

    #assert (((type(kappa) == float)
     #       | (type(kappa) == int))
      #      & (kappa > 0.)), "kappa must be positive!"
    
    assert ((type(Sigma) == np.ndarray)
            & (Sigma.shape[0] == Sigma.shape[1])), "Sigma is a square matrix!"
    
    # Find the number of dimensions using the square matrix Sigma.
    dim = Sigma.shape[0]
    
    # If the distribution is 1-D, the determinant is just the single value in
    # Sigma.
    if dim == 1:
        determinant = Sigma[0, 0]
    # Otherwise, calculate the determinant of the Sigma matrix.
    else:
        determinant = det(Sigma)
    
    # The coupled entropy calculation is broken up over several lines.
    entropy = (((np.pi/kappa)**dim) * determinant)**(kappa/(1+dim*kappa))
    entropy *= (1+dim*kappa)
    entropy *= (gamma(1/(2*kappa))/gamma(0.5*(dim + 1/kappa)))**(2*kappa
                                                                /(1+dim*kappa))
    entropy += -1
    entropy /= (2*kappa)
    
    # Return the coupled entropy.
    return entropy

def biased_coupled_probability_norm(coupled_normal, kappa, alpha):
  dim = coupled_normal.dim

  scale_mult = ((1 + dim*kappa)
    /(1 + kappa*(dim + alpha + dim*alpha*coupled_normal.kappa)))**(1/alpha)

  new_kappa = ((coupled_normal.kappa + dim*kappa*coupled_normal.kappa)
               /(1 + kappa*(dim + alpha + dim*alpha*coupled_normal.kappa)))

  new_dist = MultivariateCoupledNormal(loc=coupled_normal.loc, 
                                       scale=np.diag(coupled_normal.scale * scale_mult), 
                                      kappa=new_kappa)
  return new_dist

def coupled_probability_norm(coupled_normal,
                             kappa = 0.0, 
                             alpha = 1.0):
    
    # Return the new functions that calculates the coupled density of a value.
    return biased_coupled_probability_norm(coupled_normal, kappa, alpha).prob

def coupled_cross_entropy_norm(dist_p,
                               dist_q,
                               kappa: float = 0.0, 
                               alpha: float = 1.0, 
                               root: bool = False,
                               n=10000,
                               rounds=1,
                               seed=1) -> [float, Any]:
    
    # Fit a coupled_probability function to density_func_p with the other
    # given parameters.
    my_coupled_probability = coupled_probability_norm(dist_p,
                                                      kappa = kappa, 
                                                      alpha = alpha)
    
    dim = dist_p.dim
    
    def raised_density_func_q(x):
        return dist_q.prob(x)**(-alpha)
    
    if root == False:
        
        def no_root_coupled_cross_entropy(x):
            
            return (my_coupled_probability(x)
                    *(1/-alpha)
                    *nsc.log(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim))
        
        # Integrate the function.
        final_integration = -importance_sampling_integrator(no_root_coupled_cross_entropy, 
                                                            pdf=dist_p.prob,
                                                            sampler=dist_p.sample_n, 
                                                            n=n,
                                                            rounds=rounds,
                                                            seed=seed)
        
    else:
        def root_coupled_cross_entropy(x):

            return (my_coupled_probability(x)
                    *nsc.log(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim)**(1/alpha))
        
        # Integrate the function.
        final_integration = importance_sampling_integrator(root_coupled_cross_entropy, 
                                                           pdf=dist_p.prob,
                                                           sampler=dist_p.sample_n, 
                                                           n=n,
                                                           rounds=rounds,
                                                           seed=seed)
        
    return final_integration


def coupled_entropy_norm(dist,
                         kappa: float = 0.0, 
                         alpha: float = 1.0, 
                         root: bool = False,
                         n=10000,
                         rounds=1,
                         seed=1) -> [float, Any]:

    
    return coupled_cross_entropy_norm(dist,
                                 dist,
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 root=root,
                                 n=n,
                                 rounds=rounds,
                                 seed=seed)