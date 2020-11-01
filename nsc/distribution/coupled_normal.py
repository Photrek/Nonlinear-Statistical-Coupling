# Derived from: tensorflow_probability.distributions.student_t
# Link: https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/student_t.py
# ============================================================================
"""Coupled Normal distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import distribution, student_t
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.numeric import log1psquare


def sample_n(n, df, loc, scale, batch_shape, dtype, seed):
  """TO-DO: Daniel Svoboda
  """
  # Currently in Student-t format... 
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


def log_prob(x, df, loc, scale):
  """TO-DO: Daniel Svoboda
  """
  # Currently in Student-t format...
  y = (x - loc) * (tf.math.rsqrt(df) / scale)
  log_unnormalized_prob = -0.5 * (df + 1.) * log1psquare(y)
  log_normalization = (
      tf.math.log(tf.abs(scale)) + 0.5 * tf.math.log(df) +
      0.5 * np.log(np.pi) + tfp_math.log_gamma_difference(0.5, 0.5 * df))
  return log_unnormalized_prob - log_normalization


# class CoupledNormal(student_t.StudentT):
class CoupledNormal(distribution.Distribution):
  """Coupled Normal Distribution
  """

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='StudentT'):
    """TO-DO: Daniel Svoboda
    """
    # Currently in Student-t format... 
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, loc, scale], tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
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

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('df', 'loc', 'scale'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(df=0, loc=0, scale=0)

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _batch_shape_tensor(self, df=None, loc=None, scale=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(self.df if df is None else df),
        prefer_static.broadcast_shape(
            prefer_static.shape(self.loc if loc is None else loc),
            prefer_static.shape(self.scale if scale is None else scale)))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tf.broadcast_static_shape(self.df.shape,
                                  self.loc.shape),
        self.scale.shape)

  def _sample_n(self, n, seed=None):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, loc=loc, scale=scale)
    return sample_n(
        n,
        df=df,
        loc=loc,
        scale=scale,
        batch_shape=batch_shape,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, x):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    return log_prob(x, df, loc, scale)
