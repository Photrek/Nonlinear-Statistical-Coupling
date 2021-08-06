# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:16:39 2021

@author: jkcle
"""
from numpy.linalg import det
import numpy as np
from math import gamma

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
    
    assert (((type(kappa) == float)
            | (type(kappa) == int))
            & (kappa > 0.)), "kappa must be positive!"
    
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


scale = 1.

cauchy_entropy = np.log(4 * np.pi * scale)
coupled_entropy = coupled_normal_entropy(np.array([[scale]]), 1)

print(cauchy_entropy)
print(coupled_entropy)