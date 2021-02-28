# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 01:33:28 2021

@author: jkcle
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

batch_size = 100000
indices = np.arange(batch_size, dtype=np.int32)

dim = 1
my_func = lambda x: tf.reduce_sum(x * x, axis=1)

halton_sample = tfp.mcmc.sample_halton_sequence(dim,
                                                num_results=100,
                                                dtype=tf.float64,
                                                randomized=False)

batch_values = my_func(halton_sample)


def mc_integration(func, a, b, size=1000, seed=1):
    np.random.seed(seed)
    sample = np.random.uniform(a, b, size)
    values = func(sample)
    return np.mean(values)*(b-a)

from scipy.stats import norm

my_norm = norm(loc=0, scale=1)

print(mc_integration(my_norm.pdf, -100, 100, 10000, 1))
    