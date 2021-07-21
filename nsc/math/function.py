# -*- coding: utf-8 -*-
import numpy as np
import warnings
from typing import List

numeric_tuple = (int, float, np.float32, np.float64, np.longdouble)


def coupled_logarithm(value: [int, float, np.ndarray],
                      kappa: [int, float] = 0.0,
                      dim: int = 1
                      ) -> [float, np.ndarray]:
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
    value = np.array(value) if isinstance(value, numeric_tuple) else value
    assert isinstance(value, np.ndarray), "value must be an int, float, or np.ndarray."
    assert 0. not in value, "value must not be or contain np.ndarray zero(s)."
    if kappa == 0.:
        coupled_log_value = np.log(value)  # divide by 0 if x == 0
    else:
        coupled_log_value = (1. / kappa) * (value**(kappa / (1. + dim*kappa)) - 1.)
    return coupled_log_value


def coupled_exponential(value: [int, float, np.ndarray],
                        kappa: float = 0.0,
                        dim: int = 1
                        ) -> [float, np.ndarray]:
    """
    Generalization of the exponential function.

    Parameters
    ----------
    value : [float, np.ndarray]
        Input values in which the coupled exponential is applied to.
    kappa : float,
        Coupling parameter which modifies the coupled exponential function. 
        The default is 0.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    float
        The coupled exponential values.

    """
    #Temporarily turn off warnings for invalid powers
    warnings.simplefilter('ignore')
    
    # convert number into np.ndarray to keep consistency
    isinstance(value, (int, float, ))
    value = np.array(value) if isinstance(value, numeric_tuple) else value
    assert isinstance(value, np.ndarray), "value must be an int, float, or np.ndarray."
    # assert 0 not in value, "value must not be or contain np.ndarray zero(s)."
    
    assert isinstance(dim, int) and dim >= 0, "dim must be an integer greater than or equal to 0."
    # check that -1/d <= kappa
    # assert -1/dim <= kappa, "kappa must be greater than or equal to -1/dim."

    if kappa == 0:
        # Does not have to be vectorized
        coupled_exp_value = np.exp(value)
    else:
        #coupled_exp_value = np.vectorize(_coupled_exponential_scalar)(value, kappa, dim)
        
        #Positive base, function operates as normal 
        condition_1 = (1 + kappa*value) > 0
        
        #Negative base and positive exponent should return 0
        condition_2 = ((1 + kappa*value) <= 0) & (((1 + dim*kappa)/kappa) > 0)
        
        coupled_exp_value = np.where(condition_1, (1 + kappa*value)**((1+dim*kappa)/kappa), float('inf'))
        coupled_exp_value = np.where(condition_2, 0, coupled_exp_value)
    
    #Turn warnings back on
    warnings.simplefilter('default')
    
    return coupled_exp_value

def coupled_product(value: List[float],
                    kappa: float=0.0,
                    dims: int=1) -> float:
    """
    Coupled product function
    
    Parameters
    ----------
    value: List[float]
        The values to which the coupled product function is applied to.
        Usually a list of probabilities.
    kappa : float,
        Coupling parameter which modifies the coupled product function. 
        The default is 0.0.
    dims : 
        The dimensionality of the inputs when viewed as probability distributions.
        The default is 1 for all inputs.
        Can accept a list of dims but needs to be the same length as value.

    Returns
    -------
    float
        The result of the coupled product function.

    """
    
    #Scalar input for dims
    if type(dims) ==int:
        dims = [dims]*len(value)
    else:
        assert len(value)==len(dims), "value and dims must have the same length!"
    
    #Dim input for outer exponent should be equal to sum of dims
    D = np.sum(dims)
        
    exponent_temp = []
    
    #Calculating coupled_logarithm for each input
    for val, dim in zip(value, dims):
        log_out = coupled_logarithm(value=val, kappa=kappa, dim=dim)
        exponent_temp.append(log_out)
    
    #Summing the inputs to be fed into the coupled_exponential
    exponent = np.sum(exponent_temp)    
    
    coupled_product_value = coupled_exponential(value=exponent, kappa=kappa, dim=int(D))
    
    return coupled_product_value

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

# inner function that takes in the value on a scalar-by-sclar basis
'''
def _coupled_exponential_scalar(value, kappa, dim):
    if (1 + kappa*value) > 0:
        return (1 + kappa*value)**((1 + dim*kappa)/kappa)
    elif ((1 + dim*kappa)/kappa) > 0:
        return 0.
    else:
        return float('inf')
'''