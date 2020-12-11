# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:30:48 2020

@author: jkcle
"""
import numpy as np
import typing

def mc_integrator(integrand, a: float, b: float, n: int, seed: int = 1) -> float:
    """
    This function takes in another function to integrate from a to b using
    Monte Carlo integration using n random draws from the inputted function.
    
    Inputs
    ----------
    integrand : Input function that is being integrated.
    a : Lower bound of the integral being approximated.
    b : Upper bound of the integral being approximated.
    n : Number of draws from the integrand to be used for the approximation.
    seed : Random seed used for the random number generator.
    
    Output
    ----------
    approx_integral : The approximation of the integral from a to b.
    """
    # Set a random seed for reproducibility.
    np.random.seed(seed)
    # Draw n numbers from a Uniform(a, b) distribution.
    u = np.random.uniform(low=a, high=b, size=n)
    # Apply the integrand function to the random draws.
    rand_draws = integrand(u)
    # Approximate the integral of the integrand on the interval from a to b.
    approx_integral = (b - a) * np.sum(rand_draws)/(len(rand_draws)-1)
    # Return the approximate integral.
    return approx_integral