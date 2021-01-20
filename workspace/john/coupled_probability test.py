# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:08:25 2021

@author: jkcle
"""
from function_john import CoupledNormal
from coupled_probability_quad_int import coupled_probability
import numpy as np
import matplotlib.pyplot as plt

loc, scale = 0, 1
kappa, alpha, dim = 1, 1, 1

support = (-np.inf, np.inf)

scales_from_loc = 20
realized_support_size = 1000
realized_support_range = (loc-scales_from_loc*scale, loc+scales_from_loc*scale)
realized_support = np.linspace(realized_support_range[0], 
                               realized_support_range[1], 
                               realized_support_size)

coupled_normal = CoupledNormal(loc=loc,
                               scale=scale,
                               kappa=kappa,
                               alpha=alpha)

plt.plot(realized_support, coupled_normal.prob(realized_support))
plt.title(f'Coupled Gaussian {loc, scale} with kappa = {kappa} and alpha = {alpha}')
plt.xlabel('Realization')
plt.ylabel('Density')
plt.show()

coupled_prob_normal = coupled_probability(coupled_normal.prob, 
                                          realized_support=realized_support,
                                          kappa=kappa, 
                                          alpha=alpha, 
                                          dim=dim, 
                                          support=support)

plt.plot(realized_support, coupled_prob_normal)
plt.title(f'Coupled Probability of Coupled Gaussian {loc, scale} with kappa = {kappa} and alpha = {alpha}')
plt.xlabel('Realization')
plt.ylabel('Density')
plt.show()