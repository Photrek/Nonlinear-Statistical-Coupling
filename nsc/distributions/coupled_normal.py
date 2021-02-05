# -*- coding: utf-8 -*-
import numpy as np
from typing import List
from scipy.special import beta, gamma
from ..util.function import coupled_exponential


class CoupledNormal:
    """Coupled Normal Distribution.

    This distribution has parameters: location `loc`, 'scale', coupling `kappa`,
    and `alpha`.

    """
    def __init__(self,
                 loc: [int, float, List, np.ndarray],
                 scale: [int, float, List, np.ndarray],
                 kappa: [int, float] = 0.,
                 alpha: int = 2,
                 validate_args: bool = True
                 ):
        loc = np.asarray(loc) if isinstance(loc, List) else loc
        scale = np.asarray(scale) if isinstance(scale, List) else scale
        if validate_args:
            assert isinstance(loc, (int, float, np.ndarray)), "loc must be either an int/float type for scalar, or an list/ndarray type for multidimensional."
            assert isinstance(scale, (int, float, np.ndarray)), "scale must be either an int/float type for scalar, or an list/ndarray type for multidimensional."
            assert type(loc) == type(scale), "loc and scale must be the same type."
            if isinstance(loc, np.ndarray):
                assert loc.shape == scale.shape, "loc and scale must have the same dimensions (check respective .shape())."
                assert np.all((scale >= 0)), "All scale values must be greater or equal to 0."            
            else:
                assert scale >= 0, "scale must be greater or equal to 0."            
            assert isinstance(kappa, (int, float)), "kappa must be an int or float type."
            assert isinstance(alpha, int), "alpha must be an int that equals to either 1 or 2."
            assert alpha in [1, 2], "alpha must be equal to either 1 or 2."
        self.loc = loc
        self.scale = scale
        self.kappa = kappa
        self.alpha = alpha
        self.dim = self._n_dim()

    def _n_dim(self):
        return 1 if self._event_shape() == [] else self._event_shape()[0]

    def _batch_shape(self) -> List:
        if self._rank(self.loc) == 0:
            # return [] signifying single batch of a single distribution
            return []
        else:
            # return the batch shape in list format
            return list(self.loc.shape)

    def _event_shape(self) -> List:
        # For univariate Coupled Normal distribution, event shape is always []
        # [] signifies single random variable dim (regardless of batch size)
        return []

    def _rank(self, value: [int, float, np.ndarray]) -> int:
        # specify the rank of a given value, with rank=0 for a scalar and rank=ndim for an ndarray
        if isinstance(value, (int, float)):
            return 0 
        else:
            return len(value.shape)

    def prob(self, X: [List, np.ndarray]) -> np.ndarray:
        # Check whether input X is valid
        X = np.asarray(X) if isinstance(X, List) else X
        assert isinstance(X, np.ndarray), "X must be a List or np.ndarray."
        # assert type(X[0]) == type(self.loc), "X samples must be the same type as loc and scale."
        if isinstance(X[0], np.ndarray):
            assert X[0].shape == self.loc.shape, "X samples must have the same dimensions as loc and scale (check respective .shape())."
        # Calculate PDF with input X
        X_norm = (X-self.loc)**2 / self.scale**2
        norm_term = self._normalized_term()
        p = (coupled_exponential(X_norm, self.kappa))**-0.5 / norm_term
        return p

    # Normalization of 1-D Coupled Gaussian (NormCG)
    def _normalized_term(self) -> [int, float, np.ndarray]:
        base_term = np.sqrt(2*np.pi) * self.scale
        norm_term = base_term*self._normalization_function()
        return norm_term

    def _normalization_function(self):
        k, d = self.kappa, self.dim
        assert -1/d < k, "kappa must be greater than -1/dim."
        if k == 0:
            return 1
        elif k > 0:
            beta_input_x = 1/(2*k) + 1
            beta_input_y = d/2
            gamma_input = d/2
            return ((1 + d*k)/(2*k)**(d/2)) * beta(beta_input_x, beta_input_y)/gamma(gamma_input)
        else:  # -1 < self.kappa < 0:
            beta_input_x = (1 + d*k)/(-2*k) + 1
            beta_input_y = d/2
            gamma_input = d/2
            return (1/(2*k)**(-d/2)) * beta(beta_input_x, beta_input_y)/gamma(gamma_input)

    def __repr__(self) -> str:
        return f"<nsc.distributions.{self.__class__.__name__} batch_shape={str(self._batch_shape())} event_shape={str(self._event_shape())}>"
