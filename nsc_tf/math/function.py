# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

#numeric_tuple = (int, float, np.float32, np.float64, np.float128)
numeric_tuple = (int, float, np.longdouble)


def coupled_logarithm(value: [int, float, np.ndarray, tf.Tensor],
                      kappa: [int, float] = 0.0,
                      dim: int = 1
                      ) -> [float, np.ndarray]:
    """
    Generalization of the logarithm function, which defines smooth
    transition to power functions.

    Parameters
    ----------
    value : Input variable in which the coupled logarithm is applied to.
            Accepts int, float, np.ndarray and tf.Tensor data types.
    kappa : Coupling parameter which modifies the coupled logarithm function.
            Accepts int and float data types.
    dim : The dimension (or rank) of value. If value is scalar, then dim = 1.
          Accepts only int data type.
    """
    # convert value into np.ndarray (if scalar) to keep consistency
    value = np.array(value) if isinstance(value, numeric_tuple) else value
    #assert isinstance(value, np.ndarray), "value must be an int, float, or np.ndarray."
    
    #Tensorflow implementation
    if isinstance(value, tf.Tensor):
        #Check if values contain 0
        #assert tf.reduce_all(tf.not_equal(value,0)).numpy(), "value must not be or contain zero(s)."
        log = tf.math.log
    else:
        #General check
        assert 0. not in value, "value must not be or contain zero(s)."
        log = np.log
        
    if kappa == 0.:
        coupled_log_value = log(value)  # divide by 0 if x == 0
    else:
        coupled_log_value = (1. / kappa) * (value**(kappa / (1. + dim*kappa)) - 1.)
    return coupled_log_value


def coupled_exponential(value: [int, float, np.ndarray, tf.Tensor],
                        kappa: float = 0.0,
                        dim: int = 1
                        ) -> [float, np.ndarray]:
    """
    Generalization of the exponential function.

    Parameters
    ----------
    value : [float, np.ndarray, tf.Tensor]
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
    # convert number into np.ndarray to keep consistency
    isinstance(value, (int, float, ))
    #value = np.array(value) if isinstance(value, numeric_tuple) else value
    #assert isinstance(value, np.ndarray, tf.Tensor), "value must be an int, float, or np.ndarray."
    # assert 0 not in value, "value must not be or contain np.ndarray zero(s)."
    assert isinstance(dim, int) and dim >= 0, "dim must be an integer greater than or equal to 0."
    # check that -1/d <= kappa
    # assert -1/dim <= kappa, "kappa must be greater than or equal to -1/dim."

    if kappa == 0:
        # Does not have to be vectorized
        coupled_exp_value = np.exp(value)
    else:        
        #Positive base, function operates as normal 
        condition_1 = (1 + kappa*value) > 0
        
        #Negative base and positive exponent should return 0
        condition_2 = ((1 + kappa*value) <= 0) & (((1 + dim*kappa)/kappa) > 0)
        
        #Where function used depends on input type, if tensor, use tf.where
        if isinstance(value, tf.Tensor):
            where = tf.where
        
        #Otherwise use the numpy version
        else:
            where = np.where
            
        coupled_exp_value = where(condition_1, (1 + kappa*value)**((1+dim*kappa)/kappa), float('inf'))
        coupled_exp_value = where(condition_2, 0, coupled_exp_value)

    return coupled_exp_value

def generalized_mean(values: tf.Tensor, r: float = 1.0, weights: tf.Tensor= None) -> tf.Tensor:
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

    assert type(values) == tf.Tensor, "values must be a 1-D tensorflow Tensor."
    if len(values.shape) != 1:
        assert ((len(values.shape) == 2) 
                & ((values.shape[0] == 1)
                  | (values.shape[1] == 1))), "values must be a 1-D tensorflow Tensor"
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