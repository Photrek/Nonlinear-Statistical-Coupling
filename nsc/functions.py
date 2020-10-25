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


def normCG(sigma, kappa):
    if kappa == 0:
	result = math.sqrt(2*math.pi) * sigma
    elif kappa < 0:
	result = math.sqrt(math.pi) * sigma * math.gamma((-1+kappa) / (2*kappa)) / float(math.sqrt(-1*kappa) * math.gamma(1 - (1 / (2*kappa))))
    else:
	result = math.sqrt(math.pi) * sigma * math.gamma(1 / (2*kappa)) / float(math.sqrt(kappa) * math.gamma((1+kappa)/(2*kappa)))
  
    return result


def CoupledExpotentialDistribution(x, kappa, mean, sigma):
      coupledExponentialDistributionResult = []
      if kappa >= 0:
	input = [mean:(20*mean - mean)/10000:20*mean]
      else:
        input = [mean:(-1*sigma / kappa)/10000:(-1*sigma / kappa) + mean]
      for x in input:
        coupledExponentialDistributionResult.append((1 / sigma)*(1 / CoupledExponential((x - mu) / sigma), kappa, 1))

      return coupledExponentialDistributionResult


def MultivariateCoupledExpotentialDistribution(x, k, mu, sigma):
    pass


def CoupledNormalDistribution(x, sigma, std, kappa, alpha):
    """
    Short description
    ----------
    x :
    k :
    d :
    """

    assert std >= 0, "std must be greater than or equal to 0."
    assert alpha in [1, 2], "alpha must be set to either 1 or 2."

    coupledNormalDistributionResult = []
    if kappa >= 0:
	input = [mean*-20:(20*mean - -20*mean)/10000:mean*20]
    else:
	input = [mean - ((-1*sigma**2) / kappa)**0.5:(mean + ((-1*sigma**2) / kappa)**0.5 - mean - ((-1*sigma**2) / kappa)**0.5)/10000:mean + ((-1*sigma**2) / kappa)**0.5]
 
    normCGvalue = 1 / float(normCG(sigma, kappa))
    for i in input:
	coupledNormalDistributionResult.append(normCGvalue * (coupledExponential((x - mean)**2/sigma**2, kappa)) ** -0.5)
  
    return coupledNormalDistributionResult


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
