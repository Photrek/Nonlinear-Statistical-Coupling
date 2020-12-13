# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from typing import Any, List  # for NDArray types
#from distribution.multivariate_coupled_normal import MultivariateCoupledNormal


def coupled_logarithm(value: [float, Any], kappa: float = 0.0, dim: int = 1) -> [float, Any]:
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
    if isinstance(value, float):    
        assert value >= 0, "x must be greater or equal to 0."  # Greater than 0?????
    else:
        assert isinstance(value, np.ndarray), "x must be a np.ndarray type if a sequence, or a float if a scalar."
        # assert np.all(x), "all values in x must be "

    if kappa == 0:
        coupled_log_value = np.log(value)  # divide by 0 if x == 0
    else:
        risk_bias = kappa / (1 + dim*kappa)  # risk bias ratio
        coupled_log_value = (1 / kappa) * (value**risk_bias - 1)
    return coupled_log_value


def coupled_exponential(value: float, kappa: float = 0.0, dim: int = 1) -> float:
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
        coupled_exp_value = math.exp(value)
    else:
        risk_bias = kappa / (1 + dim*kappa)  # risk bias ratio    
        if kappa > 0:
        	coupled_exp_value = (1 + kappa*value)**(1/risk_bias) # removed negative sign and added reciprocal
        # now given that kappa < 0
        elif (1 + kappa*value) >= 0:
       		coupled_exp_value = (1 + kappa*value)**(1/risk_bias) # removed negative sign and added reciprocal
        elif (risk_bias) > 0: # removed negative sign
       		coupled_exp_value = 0
        else:
       		coupled_exp_value = float('inf')
        # else:
        # 	print("Error: kappa = 1/d is not greater than -1.")
    return coupled_exp_value


def coupled_probability(dist, kappa, alpha, d): # x, xmin, xmax): #(value: float, kappa: float = 0.0, dim: int = 1)):
    kMult = (-alpha * kappa) / (1 + d*kappa)  ## Risk bias
    new_dist_temp = [x ** (1-kMult) for x in dist]
    division_factor = np.trapz(new_dist_temp)
    new_dist = [x / division_factor for x in new_dist_temp]

    return new_dist


def coupled_entropy(dist, kappa, alpha, d, root): # x, xmin, xmax):
    if root == False:
        dist_temp = coupled_probability(dist, kappa, alpha, d)
        coupled_logarithm_values = []
        for i in dist:
            coupled_logarithm_values.append(coupled_logarithm(i**(-alpha), kappa, d))

        pre_integration = [x*y*(-1/alpha) for x,y in zip(dist_temp, coupled_logarithm_values)]
        final_integration = -1*np.trapz(pre_integration)
    else:
        dist_temp = coupled_probability(dist, kappa, alpha, d)
        coupled_logarithm_values = []
        for i in dist:
            coupled_logarithm_values.append(coupled_logarithm(i**(-alpha), kappa, d)**(1/alpha))

        pre_integration = [x * y for x, y in zip(dist_temp, coupled_logarithm_values)]
        final_integration = np.trapz(pre_integration)

    return final_integration


def norm_CG(sigma, kappa):
    if kappa == 0:
        result = math.sqrt(2*math.pi) * sigma
    elif kappa < 0:
        result = math.sqrt(math.pi) * sigma * math.gamma((-1+kappa) / (2*kappa)) / float(math.sqrt(-1*kappa) * math.gamma(1 - (1 / (2*kappa))))
    else:
        result = math.sqrt(math.pi) * sigma * math.gamma(1 / (2*kappa)) / float(math.sqrt(kappa) * math.gamma((1+kappa)/(2*kappa)))
  
    return result


def norm_multi_coupled(std: [float, Any],  kappa: float = 0.0, alpha: int = 2
                       ) -> [float, Any]:

    assert alpha == 1 or alpha == 2, "alpha must be an int and equal to either" + \
                                     " 1 (Pareto) or 2 (Gaussian)."
    
    dim = 1 if isinstance(std, float) else len(std[0])
    if alpha == 1:
        input = (std**0.5)/(1 + (-1 + dim)*kappa)
    else:  # alpha == 2
        gamma_num = math.gamma((1 + (-1 + dim)*kappa)/(2*kappa))
        gamma_dem = math.gamma((1 + dim*kappa)/(2*kappa))
        input = (((math.sqrt(math.pi)*std**0.5)*gamma_num) / (math.sqrt(kappa)*gamma_dem)) 
    return input  # currently wrong ...


def coupled_product(x):
    pass


def coupled_sum(x):
    pass


def coupled_sine(x):
    pass


def coupled_cosine(x):
    pass


def weighted_generalized_mean(x):
    pass


def coupled_box_muller(x):
    pass



'''
def CoupledNormalDistribution(x, sigma, std, kappa, alpha):

    pass
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
 '''


'''
def MultivariateCoupledNormalDistribution(mean: [float, Any], std: [float, Any], 
                                          kappa: float = 0.0, alpha: int = 2
                                          ) -> [float, Any]:
    """
    Short description
    
    Inputs
    ----------
    x : Input variable in which the coupled logarithm is applied to.
    mean : 
    std : 
    kappa : Coupling parameter which modifies the coupled logarithm function.
    alpha : Type of distribution. 1 = Pareto, 2 = Gaussian.
    """
    assert type(mean) is type(std), "mean and std must be the same type."
    if isinstance(mean, np.ndarray) and isinstance(std, np.ndarray):
        assert mean.shape == std.shape, "mean and std must have the same dim."
    assert alpha == 1 or alpha == 2, "alpha must be an int and equal to either" + \
                                     " 1 (Pareto) or 2 (Gaussian)."
  '''



