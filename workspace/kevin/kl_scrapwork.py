# -*- coding: utf-8 -*-
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v2 as tf
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions.normal import Normal
tfd = tfp.distributions


_DIVERGENCES = {}


# Normal, Normal
# @kullback_leibler.RegisterKL(Normal, Normal)
def _kl_normal_normal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Normal.
  Args:
    a: instance of a Normal distribution object.
    b: instance of a Normal distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_normal_normal'`).
  Returns:
    kl_div: Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_normal_normal'):
    b_scale = tf.convert_to_tensor(b.scale)  # We'll read it thrice.
    diff_log_scale = tf.math.log(a.scale) - tf.math.log(b_scale)
    return (
        0.5 * tf.math.squared_difference(a.loc / b_scale, b.loc / b_scale) +
        0.5 * tf.math.expm1(2. * diff_log_scale) -
        diff_log_scale)


# StudentT, StudentT
# @kullback_leibler.RegisterKL(Normal, Normal)
def _kl_student_student(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Normal.
  Args:
    a: instance of a Normal distribution object.
    b: instance of a Normal distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_normal_normal'`).
  Returns:
    kl_div: Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_normal_normal'):
    b_scale = tf.convert_to_tensor(b.scale)  # We'll read it thrice.
    diff_log_scale = tf.math.log(a.scale) - tf.math.log(b_scale)
    return (
        0.5 * tf.math.squared_difference(a.loc / b_scale, b.loc / b_scale) +
        0.5 * tf.math.expm1(2. * diff_log_scale) -
        diff_log_scale)

# CoupledNormal, CoupledNormal


# MultivariateCoupledNormal, MultivariateCoupledNormal



def _cross_entropy(ref, other, allow_nan_stats=True, name=None):
    with tf.name_scope(name or "cross_entropy"):
        return ref.entropy() + _kl_normal_normal(ref, other)


def _registered_kl(type_a, type_b):
    """Get the KL function registered for classes a and b."""
    hierarchy_a = tf_inspect.getmro(type_a)
    hierarchy_b = tf_inspect.getmro(type_b)
    dist_to_children = None
    kl_fn = None
    for mro_to_a, parent_a in enumerate(hierarchy_a):
        for mro_to_b, parent_b in enumerate(hierarchy_b):
          candidate_dist = mro_to_a + mro_to_b
          candidate_kl_fn = _DIVERGENCES.get((parent_a, parent_b), None)
          if not kl_fn or (candidate_kl_fn and candidate_dist < dist_to_children):
            dist_to_children = candidate_dist
            kl_fn = candidate_kl_fn
    return kl_fn


def kl_divergence(distribution_a, distribution_b,
                  allow_nan_stats=True, name=None):
    """Get the KL-divergence KL(distribution_a || distribution_b).
    If there is no KL method registered specifically for `type(distribution_a)`
    and `type(distribution_b)`, then the class hierarchies of these types are
    searched.
    If one KL method is registered between any pairs of classes in these two
    parent hierarchies, it is used.
    If more than one such registered method exists, the method whose registered
    classes have the shortest sum MRO paths to the input types is used.
    If more than one such shortest path exists, the first method
    identified in the search is used (favoring a shorter MRO distance to
    `type(distribution_a)`).
    Args:
      distribution_a: The first distribution.
      distribution_b: The second distribution.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    Returns:
      A Tensor with the batchwise KL-divergence between `distribution_a`
      and `distribution_b`.
    Raises:
      NotImplementedError: If no KL method is defined for distribution types
        of `distribution_a` and `distribution_b`.
    """
    kl_fn = _registered_kl(type(distribution_a), type(distribution_b))
    if kl_fn is None:
        raise NotImplementedError(
            "No KL(distribution_a || distribution_b) registered for distribution_a "
            "type {} and distribution_b type {}".format(
                type(distribution_a).__name__, type(distribution_b).__name__))
    
    name = name or "KullbackLeibler"
    with tf.name_scope(name):
        # pylint: disable=protected-access
        with distribution_a._name_and_control_scope(name + "_a"):
            with distribution_b._name_and_control_scope(name + "_b"):
                kl_t = _kl_normal_normal(distribution_a, distribution_b, name=name)
                # kl_t = kl_fn(distribution_a, distribution_b, name=name)
                if allow_nan_stats:
                    return kl_t
    
    # Check KL for NaNs
    kl_t = tf.identity(kl_t, name="kl")

    with tf.control_dependencies([
        tf.debugging.Assert(
            tf.logical_not(tf.reduce_any(tf.math.is_nan(kl_t))),
            [("KL calculation between {} and {} returned NaN values "
              "(and was called with allow_nan_stats=False). Values:".format(
                  distribution_a.name, distribution_b.name)), kl_t])
    ]):
        return tf.identity(kl_t, name="checked_kl")


if __name__=="__main__":
    # input data
    x = np.arange(-10.0, 10.0, 0.001)
    # Normal
    # <tf.Tensor: shape=(), dtype=float32, numpy=0.12768734>
    dist1 = tfd.Normal(loc=0., scale=2.)
    dist2 = tfd.Normal(loc=0., scale=3.)
    dist1.log_prob(x)
    dist1.prob(x)
    dist1.entropy()
    dist2.entropy()
    dist1.kl_divergence(dist2)
    dist1.cross_entropy(dist2)
    _kl_normal_normal(dist1, dist2)
    _cross_entropy(dist1, dist2)
    # StudentT
    # NotImplementedError: No KL(distribution_a || distribution_b) registered for distribution_a type StudentT and distribution_b type StudentT
    dist3 = tfd.StudentT(df=10, loc=0, scale=1)
    dist4 = tfd.StudentT(df=20, loc=0, scale=1)
    dist3.entropy()
    dist4.entropy()
    # dist3.kl_divergence(dist4)
    # dist3.cross_entropy(dist4)
