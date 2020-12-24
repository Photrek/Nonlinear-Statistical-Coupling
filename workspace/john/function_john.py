# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 02:15:16 2020

@author: jkcle
"""
# -*- coding: utf-8 -*-
import numpy as np
from typing import Any, List  # for NDArray types

def generalized_mean(values: np.ndarray, r: float = 1.0, weights: np.ndarray = None) -> float:
    """
    This function calculates the generalized mean of a 1-D array of non- 
    negative real numbers using the coupled logarithm and exponential functions.
    
    Parameters
    ----------
    values : np.ndarray
        DESCRIPTION : A 1-D numpy array (row vector) of non-negative numbers
         for which we are calculating the generalized mean.
    r : float, optional
        DESCRIPTION : The risk bias and the power of the generalized mean. 
        The default is 1.0 (Arithmetric Mean).
    weights : np.ndarray, optional
        DESCRIPTION : A 1-D numpy array of the weights for each value. 
        The default is None, which triggers a conditional to use equal weights.

    Returns gen_mean
    -------
    float
        DESCRIPTION : The coupled generalized mean.
    """
    
    assert type(values) == np.ndarray, "values must be a 1-D numpy ndarray."
    if len(values.shape) != 1:
        assert ((len(values.shape) == 2) 
                & ((values.shape[0] == 1)
                  | (values.shape[1] == 1))), "values must be a 1-D numpy ndarray."
    assert (values <= 0).sum() == 0, "all numbers in values must be greater than 0."
    assert ((type(r) == int) | (type(r) == float) | (type(r) == np.int32 ) 
            | (type(r) == np.float32) | (type(r) == np.int64) 
            | (type(r) == np.float64)), "r must be a numeric data type, like a float or int."
    assert ((type(weights) == type(None))
            | (type(weights) == np.ndarray)), "weights must either be None or 1-D numpy ndarray."
            
    # If weights equals None, equally weight all observations.
    if type(weights) == type(None):
        weights = weights or np.ones(len(values))
    
    # Calculate the log of the generalized mean by taking the dot product of the
    # weights vector and the vector of the coupled logarithm of the values and
    # divide the result by the sum of the the weights.
    log_gen_mean = np.dot(weights, coupled_logarithm(values, kappa=r, dim=0)) / np.sum(weights)
        
    # Calculate the generalized mean by exponentiating the log-generalized mean.
    gen_mean = coupled_exponential(log_gen_mean, kappa=r, dim=0)
    
    # Return the generalized mean.
    return gen_mean

def coupled_logarithm(values: [float, Any], kappa: float = 0.0, dim: int = 1) -> [float, Any]:
    """
    Generalization of the logarithm function, which defines smooth
    transition to power functions.

    Parameters
    ----------
    values : [float, Any]
        Input variable in which the coupled logarithm is applied to.
    kappa : float, optional
        Coupling parameter which modifies the coupled logarithm function. 
        The default is 0.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    [float, Any]
        Returns the coupled logarithm of the values.

    """

    assert (dim >= 0) & (type(dim) == int), "dim must be a postive integer."
    
    if isinstance(values, float):    
        assert values > 0, "values must be greater than 0."
    else:
        assert isinstance(values, np.ndarray), ("x must be a np.ndarray type "
                                                "if a sequence, or a float if "
                                                "a scalar.")
    
    # If kappa is 0, return the natural logarithm of the values.
    if kappa == 0:
        coupled_log_value = np.log(values)  # divide by 0 if x == 0
    # Otherwise, return (1 / kappa) * (values^(kappa / (1 + dim*kappa)) - 1).
    else:
        coupled_log_value = (1 / kappa) * (values**(kappa / (1 + dim*kappa))-1)
        
    return coupled_log_value


def coupled_exponential(values: [float, Any], kappa: float = 0.0, dim: int = 1) -> [float, Any]:
    """
    Generalization of the exponential function.

    Parameters
    ----------
    values : [float, Any]
        Input values in which the coupled exponential is applied to.
    kappa : float, optional
        Coupling parameter which modifies the coupled exponential function. 
        The default is 0.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    float
        The coupled exponential values.

    """
    
    assert (dim >= 0) & (type(dim) == int), "dim must be a postive integer."
    
    # If kappa is 0, raise e to the power of each of the values.
    if kappa == 0:
        coupled_exp_value = np.exp(values)
    # If kappa is not 0, execute the conditionals below.   
    else:
        # If kappa is positive, raise 1 + kappa*values to the power of
        # 1 / (kappa / (1 + dim*alpha))
        if kappa > 0:
        	coupled_exp_value = ((1 + kappa*values)**(1/(kappa / 
                                                    (1 + dim*kappa))))
        # If kappa is negative, but 1 + kappa*values is non-negative, raise
        # 1 + kappa*values to the power of 1 / (kappa / (1 + dim*alpha))
        elif (1 + kappa*values) >= 0:
                coupled_exp_value = (1+kappa*values)**(1/(kappa/(1+dim*kappa)))
        # If kappa is negative, 1 + kappa*values is negative, and 
        # (kappa / (1 + dim*alpha)) is positive, return 0.
        elif (kappa / (1 + dim*kappa)) > 0: 
                coupled_exp_value = 0
        # Otherwise, return infinity.
        else:
       		coupled_exp_value = float('inf')

    return coupled_exp_value


def coupled_probability(dist, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    [float, Any]
        DESCRIPTION.

    """
    
    # Calculate the risk-bias.
    kMult = (-alpha * kappa) / (1 + dim*kappa)
    # Raise the distribution densities to 1 - the risk-bias
    new_dist_temp = dist ** (1-kMult)
    # Forget dist inside the function to free up memory.
    del dist
    # Calculate the normalization factor to the coupled CDF equals 1.
    division_factor = np.trapz(new_dist_temp, dx=dx)
    # Calculate the coupled densities
    coupled_dist = new_dist_temp / division_factor

    return coupled_dist


def coupled_cross_entropy(dist_p, dist_q, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    [float, Any]
        DESCRIPTION.

    """
    
    # Raise the distrubtion P to the power (-alpha*kappa)/(1+d*kapaa) and normalize it. 
    dist_p_temp = coupled_probability(dist=dist_p, 
                                      dx=dx, 
                                      kappa=kappa, 
                                      alpha=alpha, 
                                      dim=dim)
    # Forget dist_p inside the fuction to save memory.
    del dist_p
    
    # Calculate the coupled-logarithm of the values in the distribution Q raised to the
    # negative alpha power.
    coupled_logarithm_dist_q = (1/-alpha)*coupled_logarithm(values=dist_q**(-alpha), 
                                                 kappa=kappa, 
                                                 dim=dim)
    # Forget dist_q inside the fuction to save memory.
    del dist_q
    
    # Multiply the coupled-probability values of dist_p by (
    # 1/-alpha)*coupled logarithm of dist_q.
    pre_integration = np.multiply(dist_p_temp, 
                                  coupled_logarithm_dist_q)
    # Integrate the values and multiply by negative one.
    final_integration = -np.trapz(pre_integration, dx=dx)
    
    return final_integration


def coupled_entropy(dist, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    [float, Any]
        The coupled cross-entropy between dist_p and dist_q.

    """
    
    return coupled_cross_entropy(dist_p=dist, 
                                 dist_q=dist, 
                                 dx=dx,
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 dim=dim)


def coupled_divergence(dist_p, dist_q, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    [float, Any]
       The coupled divergence.

    """
    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy(dist_p=dist_p, 
                                                           dist_q=dist_q, 
                                                           dx=dx,
                                                           kappa=kappa, 
                                                           alpha=alpha, 
                                                           dim=dim)
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy(dist=dist_p, 
                                                dx=dx, 
                                                kappa=kappa, 
                                                alpha=alpha, 
                                                dim=dim)
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p