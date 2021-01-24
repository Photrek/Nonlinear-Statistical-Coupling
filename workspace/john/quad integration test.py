# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:18:49 2021

@author: jkcle
"""
import numpy as np
from scipy.integrate import quad
from function_john import CoupledNormal
import coupled_probability_quad_int as cpqi
from typing import Any, List  # for NDArray typ

loc, scale = 0, 1
kappa, alpha = 0, 1

support = (-np.inf, np.inf)
realized_support = np.linspace(-5, 5, 1000)

coupled_normal = CoupledNormal(loc=loc,
                               scale=scale,   
                               kappa=kappa,
                               alpha=alpha)
dim = coupled_normal.n_dim()

coupled_prob = cpqi.coupled_probabilityV1(coupled_normal.prob, 
                                          realized_support=realized_support,
                                          kappa=kappa, 
                                          alpha=alpha, 
                                          dim=dim,
                                          support=support)

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

my_coupled_prob = coupled_probability(coupled_normal.prob, 
                                      kappa=kappa, 
                                      alpha=alpha, 
                                      dim=dim,
                                      support=support)

coupled_prob_values = my_coupled_prob(realized_support)

assert np.all(coupled_prob_values == coupled_prob), "Arrays don't match!"















