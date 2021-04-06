# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 00:29:11 2021

@author: jkcle
"""
import nsc
from nsc.distributions import CoupledNormal, MultivariateCoupledNormal
from numpy.linalg import det
import numpy as np
from math import gamma
from typing import Any, List  # for NDArray types
import tensorflow_probability as tfp

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
    rounds : TYPE, optional
        DESCRIPTION. The default is 1.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Set a random seed.
    np.random.seed(seed)
    
    # Create a list to hold the expectations.
    expectations = []
    
    for i in range(rounds):     
        # Generate n samples from the probability distribution.
        samples = sampler(n)
        # Estimate the integral.
        expectation = tfp.monte_carlo.expectation(
            f=lambda x: function(x)/pdf(x), 
            samples=samples, 
            log_prob=lambda x: np.log(pdf(x)), 
            use_reparametrization=False, 
            axis=None, 
            keep_dims=False, 
            name=None
            )
        # Add the estimation of the integral to the list.
        expectations.append(expectation)
    # Return the mean of the expectations.   
    return np.mean(expectations)

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
    """
    

    Parameters
    ----------
    coupled_normal : TYPE
        DESCRIPTION.
    kappa : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.

    Returns
    -------
    new_dist : TYPE
        DESCRIPTION.

    """
    dim = coupled_normal.dim
    
    scale_mult = ((1 + dim*kappa)
                  /(1 + kappa*(dim + alpha 
                               + dim*alpha*coupled_normal.kappa)))**(1/alpha)
    
    new_kappa = ((coupled_normal.kappa + dim*kappa*coupled_normal.kappa)
                 /(1 + kappa*(dim + alpha + dim*alpha*coupled_normal.kappa)))
    
    new_dist = MultivariateCoupledNormal(loc=coupled_normal.loc, 
                                         scale=np.diag(coupled_normal.scale 
                                                       * scale_mult), 
                                         kappa=new_kappa)
    return new_dist

def coupled_probability_norm(coupled_normal,
                             kappa = 0.0, 
                             alpha = 2.0):
    """
    

    Parameters
    ----------
    coupled_normal : TYPE
        DESCRIPTION.
    kappa : TYPE, optional
        DESCRIPTION. The default is 0.0.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Return the new functions that calculates the coupled density of a value.
    return biased_coupled_probability_norm(coupled_normal, kappa, alpha).prob

def coupled_cross_entropy_norm(dist_p,
                               dist_q,
                               kappa: float = 0.0, 
                               alpha: float = 2.0, 
                               root: bool = False,
                               n=10000,
                               rounds=1,
                               seed=1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 2.0.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    rounds : TYPE, optional
        DESCRIPTION. The default is 1.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    [float, Any]
        DESCRIPTION.

    """
    
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
        print("Not implemented yet.")
        pass
        
    return final_integration


def coupled_entropy_norm(dist,
                         kappa: float = 0.0, 
                         alpha: float = 2.0, 
                         root: bool = False,
                         n=10000,
                         rounds=1,
                         seed=1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    rounds : TYPE, optional
        DESCRIPTION. The default is 1.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    [float, Any]
        DESCRIPTION.

    """

    return coupled_cross_entropy_norm(dist,
                                 dist,
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 root=root,
                                 n=n,
                                 rounds=rounds,
                                 seed=seed)

def coupled_divergence_norm(dist_p, 
                            dist_q, 
                            kappa: float = 0.0, 
                            alpha: float = 2.0, 
                            root: bool = False,
                            n=10000,
                            rounds=1,
                            seed=1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    rounds : TYPE, optional
        DESCRIPTION. The default is 1.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    [float, Any]
        DESCRIPTION.

    """    
    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy_norm(dist_p,
                                                                dist_q,
                                                                kappa=kappa,
                                                                alpha=alpha,
                                                                root=root,
                                                                n=n,
                                                                rounds=rounds,
                                                                seed=seed)
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy_norm(dist_p, 
                                                     kappa=kappa, 
                                                     alpha=alpha, 
                                                     root=root,
                                                     n=n,
                                                     rounds=rounds,
                                                     seed=seed)
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p