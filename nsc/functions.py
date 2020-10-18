# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math


def CoupledLogarithm(x: float, kappa: float = 0.0, dim: int = 1) -> float:
    """
    Generalization of the logarithm function, which defines smooth 
    transition to power functions.
    
    Inputs
    ----------
    x : Input variable in which the coupled logarithm is applied to.
    kappa : Coupling parameter which modifies the coupled logarithm function.
    dim : The dimension of x, or rank if x is a tensor. Not needed?
    """
    assert dim > 0, "dim must be greater than 0."
    assert x >= 0, "x must be greater or equal to 0."
    # dim = np.linalg.matrix_rank(x)
    risk_bias = kappa / (1 + dim*kappa)  # risk bias ratio
    # risk_bias = (-2*k) / (1 + k)  # Negative sign take inverse of func
    # coupled_log_value = 0
    if kappa == 0:
        coupled_log_value = np.log(x)
    else:
        coupled_log_value = (1 / kappa) * (x**risk_bias - 1)
    return coupled_log_value


def CoupledExpotential(x: float, kappa: float = 0.0, dim: int = 1) -> float:
    """
    Short description
    ----------
    x : Input variable in which the coupled exponential is applied to.
    kappa : Coupling parameter which modifies the coupled exponential function.
    dim : The dimension of x, or rank if x is a tensor.
    """
    assert dim > 0, "dim must be greater than 0."
    assert kappa >= 0 or kappa >= (-1/dim), "kappa must be greater than -1/dim."
    risk_bias = kappa / (1 + dim*kappa)  # risk bias ratio
    # coupled_exp_value = 0
    if kappa == 0:
        coupled_exp_value = math.exp(x)
    elif kappa > 0:
    	coupled_exp_value = (1 + kappa*x)**(-risk_bias)
    # now given that kappa < 0
    elif (1 + kappa*x) >= 0:
   		coupled_exp_value = (1 + kappa*x)**(-risk_bias)
    elif (-risk_bias) > 0:
   		coupled_exp_value = 0
    else:
   		coupled_exp_value = float('inf') 
    # else:
    # 	print("Error: kappa = 1/d is not greater than -1.")
    return coupled_exp_value


def CoupledExpotentialDistribution(x, k, mu, sigma):
    pass


def MultivariateCoupledExpotentialDistribution(x, k, mu, sigma):
    pass


def CoupledGaussianDistribution(x, k, mu, sigma):
    """
    Short description
    ----------
    x : 
    k : 
    d : 
    """
    pass


def MultivariateCoupledGaussianDistribution(x, k, mu, sigma):
    pass


def NormMultiCoupled(x, k, mu, sigma):
    pass


def CoupledProbability(x):
    pass


def CoupledEntropy(x):
    pass


def CoupledProduct(x):
    pass


def CoupledSum(x):
    pass


def CoupledSine(x):
    pass


def CoupledCosine(x):
    pass


def WeightedGeneralizedMean(x):
    pass


def CoupledBoxMuller(x):
    pass
