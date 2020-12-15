# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from typing import Any, List  # for NDArray types

# import distribution module from tfp
# from tensorflow_probability.python.distributions import StudentT, Pareto
import tensorflow_probability as tfp
tfd = tfp.distributions
# import helper functions from function.py
from function import coupled_logarithm, coupled_expoential, norm_CG, norm_multi_coupled


# Make this a class?
def CoupledNormal(mean, sigma, kappa, alpha):
    """
    Short description
    
    Inputs
    ----------
    x : Input variable in which the coupled logarithm is applied to.
    mean : 
    std : 
    kappa : Coupling parameter which modifies the coupled logarithm function.
    dim : The dimension of x, or rank if x is a tensor. Not needed?
    """
    
    assert sigma >= 0, "std must be greater than or equal to 0."
    assert alpha in [1, 2], "alpha must be set to either 1 or 2."

    coupledNormalResult = []
    if kappa >= 0:
        input1 = np.linspace(float(mean-10), float(mean+10), 10000)
    else:
        input1 = np.linspace(float(mean - ((-1*sigma**2) / float(kappa))**0.5), float(mean + ((-1*sigma**2) / float(kappa))**0.5), 10000)
 
    normCGvalue = 1 / float(norm_CG(sigma, kappa))
    for x in input1:
        coupledNormalResult.append(normCGvalue * (coupled_expoential((x - mean)**2/sigma**2, kappa)) ** -0.5)
  
    return input1, coupledNormalResult


def CoupledExpotentialDistribution(kappa, mean, sigma):
    coupledExponentialDistributionResult = []
    if kappa >= 0:
        input1 = np.linspace(float(mean), float(mean+10), 10000)
    else:
        input1 = np.linspace(float(mean), float(-1*sigma / kappa) + mean, 10000)
    for x in input1:
        result1 = (float(sigma)**-1)
        result2 = float((coupled_expoential(((x - mean) / sigma), kappa, 1))**-1)
        coupledExponentialDistributionResult.append(result1*result2)

    return input1, coupledExponentialDistributionResult