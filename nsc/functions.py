# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from typing import Any  # for NDArray types


def CoupledLogarithm(x: [float, Any], kappa: float = 0.0, dim: int = 1) -> float:
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
    if isinstance(x, float):    
        assert x >= 0, "x must be greater or equal to 0."  # Greater than 0?????
    else:
        assert isinstance(x, np.ndarray), "x must be a np.ndarray type if a sequence, or a float if a scalar."
        # assert np.all(x), "all values in x must be "

    if kappa == 0:
        coupled_log_value = np.log(x)  # divide by 0 if x == 0
    else:
        risk_bias = kappa / (1 + dim*kappa)  # risk bias ratio
        coupled_log_value = (1 / kappa) * (x**risk_bias - 1)
    return coupled_log_value


def CoupledExponential(x: float, kappa: float = 0.0, dim: int = 1) -> float:
    """
    Short description
    ----------
    x : Input variable in which the coupled exponential is applied to.
    kappa : Coupling parameter which modifies the coupled exponential function.
    dim : The dimension of x, or rank if x is a tensor.
    """
    assert dim >= 0, "dim must be greater than or equal 0."
        # may also want to test that dim is an integer

        # removed the requirement on kappa; although not common kappa can be less than -1/dim
    if kappa == 0:
        coupled_exp_value = math.exp(x)
    else:
        risk_bias = kappa / (1 + dim*kappa)  # risk bias ratio    
        if kappa > 0:
        	coupled_exp_value = (1 + kappa*x)**(1/risk_bias) # removed negative sign and added reciprocal
        # now given that kappa < 0
        elif (1 + kappa*x) >= 0:
       		coupled_exp_value = (1 + kappa*x)**(1/risk_bias) # removed negative sign and added reciprocal
        elif (risk_bias) > 0: # removed negative sign
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


def CoupledNormalDistribution(x, mean, std, kappa, alpha):
    """
    Short description
    ----------
    x :
    k :
    d :
    """
    assert std >= 0, "std must be greater than or equal to 0."
    assert alpha in [1, 2], "alpha must be set to either 1 or 2."


def MultivariateCoupledDistribution(x, k, mu, sigma):
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
