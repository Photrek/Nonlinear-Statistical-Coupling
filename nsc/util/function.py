# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from typing import Any, List  # for NDArray types
#from distribution.multivariate_coupled_normal import MultivariateCoupledNormal

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


def coupled_logarithm(value: [int, float, np.ndarray], kappa: [int, float] = 0.0, dim: int = 1) -> [float, np.ndarray]:
    """
    Generalization of the logarithm function, which defines smooth
    transition to power functions.
    
    Parameters
    ----------
    value : Input variable in which the coupled logarithm is applied to.
            Accepts int, float, and np.ndarray data types.
    kappa : Coupling parameter which modifies the coupled logarithm function.
            Accepts int and float data types.
    dim : The dimension (or rank) of value. If value is scalar, then dim = 1.
          Accepts only int data type.
    """
    # convert value into np.ndarray (if scalar) to keep consistency
    value = np.array(value) if isinstance(value, (int, float)) else value
    assert isinstance(value, np.ndarray), "value must be an int, float, or np.ndarray."
    assert 0. not in value, "value must not be or contain any zero(s)."
    if kappa == 0.:
        coupled_log_value = np.log(value)  # divide by 0 if x == 0
    else:
        coupled_log_value = (1. / kappa) * (value**(kappa / (1. + dim*kappa)) - 1.)
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
        if kappa > 0:
        	coupled_exp_value = (1 + kappa*value)**(1/(kappa / (1 + dim*kappa))) # removed negative sign and added reciprocal
        # now given that kappa < 0
        elif (1 + kappa*value) >= 0:
       		coupled_exp_value = (1 + kappa*value)**(1/(kappa / (1 + dim*kappa))) # removed negative sign and added reciprocal
        elif (kappa / (1 + dim*kappa)) > 0: # removed negative sign
       		coupled_exp_value = 0
        else:
       		coupled_exp_value = float('inf')
        # else:
        # 	print("Error: kappa = 1/d is not greater than -1.")
    return coupled_exp_value


def coupled_probability(dist, kappa, alpha, d): # x, xmin, xmax): #(value: float, kappa: float = 0.0, dim: int = 1)):
    kMult = (-alpha * kappa) / (1 + d*kappa)
    new_dist_temp = [x ** (1-kMult) for x in dist]
    division_factor = np.trapz(new_dist_temp)
    new_dist = [x / division_factor for x in new_dist_temp]

    return new_dist


def coupled_entropy(dist, kappa, alpha, d): # x, xmin, xmax):
    dist_temp = coupled_probability(dist, kappa, alpha, d)
    coupled_logarithm_values = []
    for i in dist:
        coupled_logarithm_values.append(coupled_logarithm(i, kappa, d))

    pre_integration = [x*y for x,y in zip(dist_temp, coupled_logarithm_values)]
    final_integration = -1*np.trapz(pre_integration)
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


