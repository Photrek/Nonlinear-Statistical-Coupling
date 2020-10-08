# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def CoupledExponential(x: float, k: float = 0.0) -> float:
    """
    Short description
    ----------
    x : 
    k : 
    d : 
    """
    # Assigned to Daniel Svoboda
    pass


def CoupledLogarithm(x: float, k: float = 0.0) -> float:
    """
    Generalization of the logarithm function, which defines smooth 
    transition to power functions.
    
    Inputs
    ----------
    x : Input variable in which the coupled logarithm is applied to.
    k : Coupling paramtere which modifies the coupled logarithm function.
    d : The dimension of x, or rank if x is a tensor. Not needed?
    """
    assert x >= 0, "x must be greater or equal to 0."
    d = np.linalg.matrix_rank(x)
    if k != 0:
        r = k / (1 + d*k)  # risk bias ??
        # r = (-2*k) / (1 + k)
        return (1 / k) * (x**r - 1)
    else:
        return np.log(x)


def CoupledNormalDistribution(x, k, mu, sigma):
    """
    Short description
    ----------
    x : 
    k : 
    d : 
    """
    pass


def CoupledExpotentialDistribution(x, k, mu, sigma):
    pass


def MultivariateCoupledDistribution(x, k, mu, sigma):
    pass


def NormMultiCoupled(x, k, mu, sigma):
    pass


def CoupledProbability(x):
    pass


def CoupledEntropy(x):
    pass


def CoupledProduct(x):
    pass


def CoupledSum(x):
    pass


def CoupledSine(x):
    pass


def CoupledCosine(x):
    pass


def WeightedGeneralizedMean(x):
    pass


def CoupledBoxMuller(x):
    pass