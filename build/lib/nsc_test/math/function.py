# -*- coding: utf-8 -*-
import numpy as np

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

    return coupled_exp_value


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