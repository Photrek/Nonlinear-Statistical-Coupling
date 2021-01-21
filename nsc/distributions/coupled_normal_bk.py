# Derived from: tensorflow_probability.distributions.student_t
# Link: https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/student_t.py
# ============================================================================
"""Coupled Normal distribution class."""

# Dependency imports
import numpy as np
import pandas as pd
from typing import List
# import tensorflow.compat.v2 as tf

# from tensorflow_probability.python import math as tfp_math
# from tensorflow_probability.python.bijectors import identity as identity_bijector
# from tensorflow_probability.python.distributions import distribution, student_t
# from tensorflow_probability.python.distributions import kullback_leibler
# from tensorflow_probability.python.internal import assert_util
# from tensorflow_probability.python.internal import distribution_util
# from tensorflow_probability.python.internal import dtype_util
# from tensorflow_probability.python.internal import prefer_static
# from tensorflow_probability.python.internal import reparameterization
# from tensorflow_probability.python.internal import samplers
# from tensorflow_probability.python.internal import tensor_util
# from tensorflow_probability.python.math.numeric import log1psquare


def sample_n(n, kappa, loc, scale, batch_shape, dtype, seed):
    """TO-DO: Kevin Chen
    """
    normal_seed, gamma_seed = samplers.split_seed(seed, salt='student_t')
    shape = tf.concat([[n], batch_shape], 0)
    
    normal_sample = samplers.normal(shape, dtype=dtype, seed=normal_seed)
    kappa = kappa * tf.ones(batch_shape, dtype=dtype)
    gamma_sample = samplers.gamma([n],
                                  0.5 / kappa,
                                  beta=0.5,
                                  dtype=dtype,
                                  seed=gamma_seed)
    samples = normal_sample * tf.math.rsqrt(gamma_sample * kappa)
    return samples * scale + loc
    '''
    StudentT code:
    normal_seed, gamma_seed = samplers.split_seed(seed, salt='student_t')
    shape = tf.concat([[n], batch_shape], 0)
    
    normal_sample = samplers.normal(shape, dtype=dtype, seed=normal_seed)
    df = df * tf.ones(batch_shape, dtype=dtype)
    gamma_sample = samplers.gamma([n],
                                  0.5 * df,
                                  beta=0.5,
                                  dtype=dtype,
                                  seed=gamma_seed)
    samples = normal_sample * tf.math.rsqrt(gamma_sample / df)
    return samples * scale + loc
    '''


def log_prob(x, kappa, loc, scale):
    """TO-DO: Kevin Chen
    """
    y = (x - loc) / (tf.math.rsqrt(kappa) * scale)
    # y = (x - loc) * (tf.math.rsqrt(1 / kappa) / scale)
    log_unnormalized_prob = -0.5 * ((1 / kappa) + 1.) * log1psquare(y)
    log_normalization = (
        tf.math.log(tf.abs(scale)) + 0.5 * tf.math.log(1 / kappa) +
        0.5 * np.log(np.pi) + tfp_math.log_gamma_difference(0.5, 0.5 / kappa))
    return log_unnormalized_prob - log_normalization
    '''
    StudentT code:
    y = (x - loc) * (tf.math.rsqrt(df) / scale)
    log_unnormalized_prob = -0.5 * (df + 1.) * log1psquare(y)
    log_normalization = (
        tf.math.log(tf.abs(scale)) + 0.5 * tf.math.log(df) +
        0.5 * np.log(np.pi) + tfp_math.log_gamma_difference(0.5, 0.5 * df))
    return log_unnormalized_prob - log_normalization
    '''


def entropy(df, scale, batch_shape, dtype):
    """TO-DO: Kevin Chen
    """
    pass
    '''
    v = tf.ones(batch_shape, dtype=dtype)
    u = v * df
    return (tf.math.log(tf.abs(scale)) + 0.5 * tf.math.log(df) +
            tfp_math.lbeta(u / 2., v / 2.) + 0.5 * (df + 1.) *
            (tf.math.digamma(0.5 * (df + 1.)) - tf.math.digamma(0.5 * df)))
    '''


# class CoupledNormal(student_t.StudentT):
class CoupledNormal(distribution.Distribution):
  """Coupled Normal Distribution.

  This distribution has parameters: coupling `kappa`, location `loc`,
  and `scale`.

  """

  def __init__(self,
               kappa,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='StudentT'):

    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([kappa, loc, scale], tf.float32)
      self._kappa = tensor_util.convert_nonref_to_tensor(
          kappa, name='kappa', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype((self._df, self._loc, self._scale))
      super(CoupledNormal, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('kappa', 'loc', 'scale'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(kappa=0, loc=0, scale=0)

  @property
  def kappa(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._kappa

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _batch_shape_tensor(self, kappa=None, loc=None, scale=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(self.kappa if kappa is None else kappa),
        prefer_static.broadcast_shape(
            prefer_static.shape(self.loc if loc is None else loc),
            prefer_static.shape(self.scale if scale is None else scale)))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tf.broadcast_static_shape(self.kappa.shape,
                                  self.loc.shape),
        self.scale.shape)

  def _sample_n(self, n, seed=None):
    kappa = tf.convert_to_tensor(self.kappa)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(kappa=kappa, loc=loc, scale=scale)
    return sample_n(
        n,
        kappa=kappa,
        loc=loc,
        scale=scale,
        batch_shape=batch_shape,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, x):
    kappa = tf.convert_to_tensor(self.kappa)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    return log_prob(x, kappa, loc, scale)

  def _entropy(self):
    kappa = tf.convert_to_tensor(self.kappa)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(kappa=kappa, scale=scale)
    return entropy(kappa, scale, batch_shape, self.dtype)

@kullback_leibler.RegisterKL(CoupledNormal, CoupledNormal)
def _kl_coupled_normal(a, b, name=None):
    """TO-DO: Kevin Chen
    """
    pass
    '''
    Normal code:
    with tf.name_scope(name or 'kl_normal_normal'):
        b_scale = tf.convert_to_tensor(b.scale)  # We'll read it thrice.
        diff_log_scale = tf.math.log(a.scale) - tf.math.log(b_scale)
        return (
            0.5 * tf.math.squared_difference(a.loc / b_scale, b.loc / b_scale) +
            0.5 * tf.math.expm1(2. * diff_log_scale) -
            diff_log_scale)
    '''