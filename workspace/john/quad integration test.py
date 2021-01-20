# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:18:49 2021

@author: jkcle
"""
import numpy as np
from scipy.integrate import quad
from function_john import CoupledNormal

loc, scale = 0, 1
alpha = 1
support = (-np.inf, np.inf)

for kappa in [0, 0.01, 0.1, 1]:
    
    coupled_normal = CoupledNormal(loc=loc,
                                   scale=scale,
                                   kappa=kappa,
                                   alpha=alpha)
    integral = quad(coupled_normal.prob, a=support[0], b=support[1])[0]
    print(f'Integral of Coupled Gaussian {loc, scale} from {support} with kappa = {kappa}: {integral}')