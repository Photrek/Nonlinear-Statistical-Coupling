# Derived from: tensorflow_probability.distributions.multivariate_student_t
# Link: https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/multivariate_student_t.py
# ============================================================================
"""Multivariate Coupled Normal distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import chi2 as chi2_lib
from tensorflow_probability.python.distributions import distribution, multivariate_student_t
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'MultivariateCoupledNormal',
]


# class MultivariateCoupledNormal(multivariate_student_t.MultivariateStudentTLinearOperator):
class MultivariateCoupledNormal(distribution.Distribution):
    """TO-DO: Kevin Chen
    
    #### Mathematical Details
    The probability density function (pdf) is,
    ```none
    pdf(x; df, loc, Sigma) = (1 + ||y||**2 / df)**(-0.5 (df + k)) / Z
    where,
    y = inv(Sigma) (x - loc)
    Z = abs(det(Sigma)) sqrt(df pi)**k Gamma(0.5 df) / Gamma(0.5 (df + k))
    ```
    where:
    * `df` is a positive scalar.
    * `loc` is a vector in `R^k`,
    * `Sigma` is a positive definite `shape' matrix in `R^{k x k}`, parameterized
       as `scale @ scale.T` in this class,
    * `Z` denotes the normalization constant, and,
    * `||y||**2` denotes the squared Euclidean norm of `y`.
    The Multivariate Student's t-distribution distribution is a member of the
    [location-scale
    family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
    constructed as,
    ```none
    X ~ MultivariateT(loc=0, scale=1)   # Identity scale, zero shift.
    Y = scale @ X + loc
    ```
    #### Examples
    ```python
    tfd = tfp.distributions
    # Initialize a single 3-variate Student's t.
    df = 3.
    loc = [1., 2, 3]
    scale = [[ 0.6,  0. ,  0. ],
             [ 0.2,  0.5,  0. ],
             [ 0.1, -0.3,  0.4]]
    sigma = tf.matmul(scale, scale, adjoint_b=True)
    # ==> [[ 0.36,  0.12,  0.06],
    #      [ 0.12,  0.29, -0.13],
    #      [ 0.06, -0.13,  0.26]]
    mvt = tfd.MultivariateStudentTLinearOperator(
        df=df,
        loc=loc,
        scale=tf.linalg.LinearOperatorLowerTriangular(scale))
    # Covariance is closely related to the sigma matrix (for df=3, it is 3x of the
    # sigma matrix).
    mvt.covariance().eval()
    # ==> [[ 1.08,  0.36,  0.18],
    #      [ 0.36,  0.87, -0.39],
    #      [ 0.18, -0.39,  0.78]]
    # Compute the pdf of an`R^3` observation; return a scalar.
    mvt.prob([-1., 0, 1]).eval()  # shape: []
    
    """
    
    def __init__(self,
                 kappa,
                 loc,
                 scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='MultivariateCoupledNormal'):
      """Construct Multivariate Student's t-distribution on `R^k`.
      The `batch_shape` is the broadcast shape between `df`, `loc` and `scale`
      arguments.
      The `event_shape` is given by last dimension of the matrix implied by
      `scale`. The last dimension of `loc` must broadcast with this.
      Additional leading dimensions (if any) will index batches.
      Args:
        kappa: 
        df: A positive floating-point `Tensor`. Has shape `[B1, ..., Bb]` where `b
          >= 0`.
        loc: Floating-point `Tensor`. Has shape `[B1, ..., Bb, k]` where `k` is
          the event size.
        scale: Instance of `LinearOperator` with a floating `dtype` and shape
          `[B1, ..., Bb, k, k]`.
        validate_args: Python `bool`, default `False`. Whether to validate input
          with asserts. If `validate_args` is `False`, and the inputs are invalid,
          correct behavior is not guaranteed.
        allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
          exception if a statistic (e.g. mean/variance/etc...) is undefined for
          any batch member If `True`, batch members with valid parameters leading
          to undefined statistics will return NaN for this statistic.
        name: The name to give Ops created by the initializer.
      Raises:
        TypeError: if not `scale.dtype.is_floating`.
        ValueError: if not `scale.is_non_singular`.
      """
      parameters = dict(locals())
      if not dtype_util.is_floating(scale.dtype):
        raise TypeError('`scale` must have floating-point dtype.')
      if validate_args and not scale.is_non_singular:
        raise ValueError('`scale` must be non-singular.')
    
      with tf.name_scope(name) as name:
        dtype = dtype_util.common_dtype([kappa, loc, scale], dtype_hint=tf.float32)
        self._kappa = tensor_util.convert_nonref_to_tensor(
            kappa, name='kappa', dtype=dtype)
        self._loc = tensor_util.convert_nonref_to_tensor(
            loc, name='loc', dtype=dtype)
        self._scale = scale
    
        super(MultivariateCoupledNormal, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            parameters=parameters,
            name=name,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats)
        self._parameters = parameters
    
    @property
    def loc(self):
      """The location parameter of the distribution.
      `loc` applies an elementwise shift to the distribution.
      ```none
      X ~ MultivariateT(loc=0, scale=1)   # Identity scale, zero shift.
      Y = scale @ X + loc
      ```
      Returns:
        The `loc` `Tensor`.
      """
      return self._loc
    
    @property
    def scale(self):
      """The scale parameter of the distribution.
      `scale` applies an affine scale to the distribution.
      ```none
      X ~ MultivariateT(loc=0, scale=1)   # Identity scale, zero shift.
      Y = scale @ X + loc
      ```
      Returns:
        The `scale` `LinearOperator`.
      """
      return self._scale
    
    @property
    def kappa(self):
      """The degrees of freedom of the distribution.
      This controls the degrees of freedom of the distribution. The tails of the
      distribution get more heavier the smaller `df` is. As `df` goes to
      infinitiy, the distribution approaches the Multivariate Normal with the same
      `loc` and `scale`.
      Returns:
        The `df` `Tensor`.
      """
      return self._kappa
    
    @property
    def inv_kappa(self):
      """The degrees of freedom of the distribution.
      This controls the degrees of freedom of the distribution. The tails of the
      distribution get more heavier the smaller `df` is. As `df` goes to
      infinitiy, the distribution approaches the Multivariate Normal with the same
      `loc` and `scale`.
      Returns:
        The `df` `Tensor`.
      """
      return 1 / self._kappa
    
    def _batch_shape_tensor(self):
      shape_list = [
          self.scale.batch_shape_tensor(),
          tf.shape(self.inv_kappa),
          tf.shape(self.loc)[:-1]
      ]
      return functools.reduce(tf.broadcast_dynamic_shape, shape_list)
    
    def _batch_shape(self):
      shape_list = [self.scale.batch_shape, self.inv_kappa.shape, self.loc.shape[:-1]]
      return functools.reduce(tf.broadcast_static_shape, shape_list)
    
    def _event_shape_tensor(self):
      return self.scale.range_dimension_tensor()[tf.newaxis]
    
    def _event_shape(self):
      return self.scale.range_dimension
    
    def _sample_shape(self):
      return tf.concat([self.batch_shape_tensor(), self.event_shape_tensor()], -1)
    
    def _sample_n(self, n, seed=None):
      # Like with the univariate Student's t, sampling can be implemented as a
      # ratio of samples from a multivariate gaussian with the appropriate
      # covariance matrix and a sample from the chi-squared distribution.
      normal_seed, chi2_seed = samplers.split_seed(seed, salt='multivariate coupled')
    
      loc = tf.broadcast_to(self.loc, self._sample_shape())
      # This needs to be fixed
      mvn = mvn_linear_operator.MultivariateNormalLinearOperator(
          loc=tf.zeros_like(loc), scale=self.scale)
      normal_samp = mvn.sample(n, seed=normal_seed)
    
      inv_kappa = tf.broadcast_to(self.inv_kappa, self.batch_shape_tensor())
      chi2 = chi2_lib.Chi2(df=inv_kappa)
      chi2_samp = chi2.sample(n, seed=chi2_seed)
    
      return (self._loc +
              normal_samp * tf.math.rsqrt(chi2_samp * self._kappa)[..., tf.newaxis])
    
    def _log_normalization(self):
      inv_kappa = tf.convert_to_tensor(self.inv_kappa)
      num_dims = tf.cast(self.event_shape_tensor()[0], self.dtype)
      return (tfp_math.log_gamma_difference(num_dims / 2., inv_kappa / 2.) +
              num_dims / 2. * (tf.math.log(inv_kappa) + np.log(np.pi)) +
              self.scale.log_abs_determinant()
              )
    
    def _log_unnormalized_prob(self, value):
      inv_kappa = tf.convert_to_tensor(self.inv_kappa)
      value = value - self._loc
      value = self.scale.solve(value[..., tf.newaxis])
    
      num_dims = tf.cast(self.event_shape_tensor()[0], self.dtype)
      mahalanobis = tf.norm(value, axis=[-1, -2])
      return -(num_dims + inv_kappa) / 2. * tfp_math.log1psquare(
          mahalanobis / tf.sqrt(inv_kappa))
    
    def _log_prob(self, value):
      return self._log_unnormalized_prob(value) - self._log_normalization()
    
    '''
    def _entropy(self):
      # df = tf.broadcast_to(self.df, self.batch_shape_tensor())
      inv_kappa = tf.broadcast_to(self.inv_kappa, self.batch_shape_tensor())
      num_dims = tf.cast(self.event_shape_tensor()[0], self.dtype)
    
      shape_factor = self._scale.log_abs_determinant()
      beta_factor = (num_dims / 2. * (tf.math.log(inv_kappa) + np.log(np.pi)) +
                     tfp_math.log_gamma_difference(num_dims / 2., inv_kappa / 2.))
      digamma_factor = (num_dims + inv_kappa) / 2. * (
          tf.math.digamma((num_dims + inv_kappa) / 2.) - tf.math.digamma(inv_kappa / 2.))
      return shape_factor + beta_factor + digamma_factor
    '''
