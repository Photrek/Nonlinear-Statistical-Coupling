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
def CoupledNormal(x, sigma, std, kappa, alpha):
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

    coupledNormalResult = []
    if kappa >= 0:
        input = [mean*-20:(20*mean - -20*mean)/10000:mean*20]
    else:
        input = [mean - ((-1*sigma**2) / kappa)**0.5:(mean + ((-1*sigma**2) / kappa)**0.5 - mean - ((-1*sigma**2) / kappa)**0.5)/10000:mean + ((-1*sigma**2) / kappa)**0.5]
 
    normCGvalue = 1 / float(normCG(sigma, kappa))
    for i in input:
        pledNormalResult.append(normCGvalue * (coupledExponential((x - mean)**2/sigma**2, kappa)) ** -0.5)
  
    return coupledNormalResult



class MultivariateCoupledNormal:
# class MultivariateCoupledNormal(distribution.Distribution):

    def __init__(self, mean, std, kappa, alpha
                 validate_args=False, allow_nan_stats=True,
                 name='MultivariateCoupledNormal'
                 ):
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

        self.dist = std.Pareto(concentration=1/kappa, scale=std) if alpha == 1 else
                    std.StudentT(df=1/kappa, loc=mean, scale=std)

        '''
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([1/kappa, mean, std], tf.float32)
            self._df = tensor_util.convert_nonref_to_tensor(
                1/kappa, name='df', dtype=dtype)
            self._loc = tensor_util.convert_nonref_to_tensor(
                mean, name='loc', dtype=dtype)
            self._scale = tensor_util.convert_nonref_to_tensor(
                std, name='scale', dtype=dtype)
            dtype_util.assert_same_float_dtype((self._df, self._loc, self._scale))
            super(StudentT, self).__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name)
        '''

    def sample(sample_shape=(), seed=None, name='sample', **kwargs):
        return self.dist.sample(sample_shape, seed, name, **kwargs)
        

class MultivariateCoupledNormal(mean: [float, Any], std: [float, Any], 
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
def CoupledExpotentialDistribution(x, kappa, mean, sigma):
    coupledExponentialDistributionResult = []
    if kappa >= 0:
        result = [mean:(20*mean - mean)/10000:20*mean]
    else:
        input = [mean:(-1*sigma / kappa)/10000:(-1*sigma / kappa) + mean]
    for x in input:
        coupledExponentialDistributionResult.append((1 / sigma)*(1 / CoupledExponential((x - mu) / sigma), kappa, 1))

    return coupledExponentialDistributionResult
'''


'''
def MultivariateCoupledExpotentialDistribution(x, k, mu, sigma):
    pass
'''