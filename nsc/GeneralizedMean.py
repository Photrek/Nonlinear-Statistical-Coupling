# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:51:59 2022

@author: jkcle

Holds the GeneralizedMean class.
"""
import pandas as pd
import tensorflow as tf


def kappa_to_r(kappa, dim=1):
  return -2*kappa / (1 + dim*kappa)
  

def replace_tensor_zeros(tnsr, power=-300):
    """Replaces 0 elements in a 1D Tensor with 10^power."""
    idx = tf.where(tf.equal(tnsr, 0))
    values = tf.pow(tf.multiply(tf.squeeze(tf.ones(idx.shape, dtype=tnsr.dtype)), 10), power)
    delta = tf.SparseTensor(indices=idx, values=values, dense_shape=tnsr.shape)
    new_tnsr = tf.add(tnsr, tf.sparse.to_dense(delta))
    return new_tnsr


def generalized_mean(values, r):
    inv_n = (1 / tf.size(values).numpy())
    new_values = replace_tensor_zeros(values)
    if r != 0:
      gen_mean = tf.pow(
          inv_n * tf.reduce_sum(
              tf.pow(new_values, r)
              ), 
              (1/r)
          )
    else:
      gen_mean = tf.exp(inv_n * tf.reduce_sum(tf.math.log(new_values)))
    return gen_mean


class GeneralizedMean:

    def __init__(self, ll_values, kl_values, kappa, z_dim):
        self.kappa = kappa
        self.z_dim = z_dim
        self.gmean_metrics = pd.Series()
        self.gmean_log_prob_values = {}
        print('\nELBO GENERALIZED MEANS')  # TODO REMOVE
        self._save_generalized_mean_metrics('elbo', ll_values - kl_values)
        print('\nRECONSTRUCTION GENERALIZED MEANS')  # TODO REMOVE
        self._save_generalized_mean_metrics('recon', ll_values)
        print('\nDIVERGENCE GENERALIZED MEANS')  # TODO REMOVE
        self._save_generalized_mean_metrics('kldiv', -kl_values)
        return
    
    def _save_generalized_mean_metrics(self, key, log_prob_values):
        # Save the generalized means and metric values
        prob_values = tf.math.exp(log_prob_values)
        self.inv_n = 1 / len(prob_values)  # 1/n
        decisiveness = self._calculate_decisiveness(prob_values)
        accuracy = self._calculate_accuracy(prob_values)
        robustness = self._calculate_robustness(prob_values)
        r = kappa_to_r(float(self.kappa), dim=1)  # TODO USE Z-DIM?
        gen_mean = generalized_mean(prob_values, r)
        curr_metrics = pd.Series(
            [decisiveness, accuracy, robustness, gen_mean],
            index=[
                f'{key}_decisiveness', f'{key}_accuracy', f'{key}_robustness',
                f'{key}_{round(r, 3)}_generalized_mean'
                ]
            )
        self.gmean_metrics = self.gmean_metrics.append(curr_metrics)
        self.gmean_log_prob_values[key] = log_prob_values
    
    def _calculate_decisiveness(self, values):
        # Decisiveness = Arithmetic mean
        result = generalized_mean(values, r=1.0)
        return result
    
    def _calculate_accuracy(self, values):
        # Accuracy = Geometric mean
        result = generalized_mean(values, r=0.0)
        return result
    
    def _calculate_robustness(self, values):
        # Robustness = -2/3 Mean
        result = generalized_mean(values, r=-2/3)
        return result
    
    def get_metrics(self):
        return self.gmean_metrics
    
    def get_log_prob_values(self):
        return self.gmean_log_prob_values
