# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from typing import Any, List  # for NDArray types

# import distribution module from tfp
# from tensorflow_probability.python.distributions import StudentT, Pareto
#import tensorflow_probability as tfp
#tfd = tfp.distributions
# import helper functions from function.py
from function import coupled_logarithm, coupled_exponential as coupledExponential, norm_CG as normCG, norm_multi_coupled


# Make this a class?
def CoupledNormal(loc: [float, Any], scale: [float, Any], kappa: [float, Any], alpha: int = 2):
    """
    Short description
    
    Inputs
    ----------
    x : Input variable in which the coupled logarithm is applied to.
    loc : 
    scale : 
    kappa : Coupling parameter which modifies the coupled logarithm function.
    dim : The dimension of x, or rank if x is a tensor. Not needed?
    """

    assert scale >= 0, "scale must be greater than or equal to 0."
    assert alpha in [1, 2], "alpha must be set to either 1 or 2."

    coupledNormalResult = []
    if kappa >= 0:
        input1 = range(loc*-20, loc*20, int((20*loc - -20*loc)/10000))
    else:
        input1 = range(loc - ((-1*scale**2) / kappa)**0.5, loc + ((-1*scale**2) / kappa)**0.5, (loc + ((-1*scale**2) / kappa)**0.5 - loc - ((-1*scale**2) / kappa)**0.5)/10000)
 
    normCGvalue = 1 / float(normCG(scale, kappa))
    for x in input1:
        coupledNormalResult.append(normCGvalue * (coupledExponential((x - loc)**2/scale**2, kappa)) ** -0.5)
  
    return coupledNormalResult


class MultivariateCoupledNormal:
# class MultivariateCoupledNormal(distribution.Distribution):

    def __init__(self, loc, scale, kappa, alpha,
                 validate_args=False, allow_nan_stats=True,
                 name='MultivariateCoupledNormal'
                 ):
        """
        Short description
        
        Inputs
        ----------
        x : Input variable in which the coupled logarithm is applied to.
        loc : 
        scale : 
        kappa : Coupling parameter which modifies the coupled logarithm function.
        alpha : Type of distribution. 1 = Pareto, 2 = Gaussian.
        """

        '''
        assert type(loc) is type(scale), "loc and scale must be the same type."
        if isinstance(loc, np.ndarray) and isinstance(scale, np.ndarray):
            assert loc.shape == scale.shape, "loc and scale must have the same dim."
        assert alpha == 1 or alpha == 2, "alpha must be an int and equal to either" + \
                                         " 1 (Pareto) or 2 (Gaussian)."

        self.dist = scale.Pareto(concentration=1/kappa, scale=scale) if alpha == 1 else
                    scale.StudentT(df=1/kappa, loc=loc, scale=scale)
        '''

        '''
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([1/kappa, loc, scale], tf.float32)
            self._df = tensor_util.convert_nonref_to_tensor(
                1/kappa, name='df', dtype=dtype)
            self._loc = tensor_util.convert_nonref_to_tensor(
                loc, name='loc', dtype=dtype)
            self._scale = tensor_util.convert_nonref_to_tensor(
                scale, name='scale', dtype=dtype)
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

'''
class MultivariateCoupledNormal(loc: [float, Any], scale: [float, Any], 
                                kappa: float = 0.0, alpha: int = 2
                                ) -> [float, Any]:
'''
"""
Short description

Inputs
----------
x : Input variable in which the coupled logarithm is applied to.
loc : 
scale : 
kappa : Coupling parameter which modifies the coupled logarithm function.
alpha : Type of distribution. 1 = Pareto, 2 = Gaussian.
"""
'''
assert type(loc) is type(scale), "loc and scale must be the same type."
if isinstance(loc, np.ndarray) and isinstance(scale, np.ndarray):
    assert loc.shape == scale.shape, "loc and scale must have the same dim."
assert alpha == 1 or alpha == 2, "alpha must be an int and equal to either" + \
                                 " 1 (Pareto) or 2 (Gaussian)."
    '''

'''
def CoupledExpotentialDistribution(x, kappa, loc, scale):
    coupledExponentialDistributionResult = []
    if kappa >= 0:
        result = [loc:(20*loc - loc)/10000:20*loc]
    else:
        input = [loc:(-1*scale / kappa)/10000:(-1*scale / kappa) + loc]
    for x in input:
        coupledExponentialDistributionResult.append((1 / scale)*(1 / CoupledExponential((x - mu) / scale), kappa, 1))

    return coupledExponentialDistributionResult
'''


'''
def MultivariateCoupledExpotentialDistribution(x, k, mu, scale):
    pass
'''