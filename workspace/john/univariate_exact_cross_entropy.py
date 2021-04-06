# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:45:56 2021

@author: jkcle
"""
from math import gamma
import numpy as np

def exact_univariate_coupled_cross_entropy(dist_p,
                                           dist_q,
                                           kappa=0.01):
    assert dist_p.kappa == dist_q.kappa, "The kappas must match!"
    # Get the means.
    mu_p, mu_q = dist_p.loc[0], dist_q.loc[0]
    # Get the scales.
    scale_p, scale_q = dist_p.scale[0][0], dist_q.scale[0][0]
    
    cpld_crss_entropy = -kappa**-1
    cpld_crss_entropy += (-np.pi**(kappa/(1+kappa))*kappa**(-kappa/(1+kappa))
                          *(2*mu_p-mu_q)*mu_q*scale_q**(-2/(1+kappa))
                          *(gamma(1/(2*kappa))/gamma((1+kappa)/(2*kappa)))
                          **(2*kappa/(1+kappa)))
    cpld_crss_entropy += ((np.pi**(kappa/(1+kappa))*kappa**(-kappa/(1+kappa))
                          *((1+2*kappa)*mu_p**2+scale_p**2)
                          *scale_q**(-2/(1+kappa))*(gamma(1/(2*kappa))
                                                    /gamma((1+kappa)
                                                           /(2*kappa)))
                          **(2*kappa/(1+kappa)))
                          /(1+2*kappa))
    cpld_crss_entropy += (np.pi**(kappa/(1+kappa))*kappa**(-(1+2*kappa)
                                                           /(1+kappa))
                          *(scale_q*gamma(1/(2*kappa))
                           /gamma((1+kappa)/(2*kappa)))**((2*kappa)/(1+kappa)))
    
    cpld_crss_entropy *= 0.5
    
    return cpld_crss_entropy