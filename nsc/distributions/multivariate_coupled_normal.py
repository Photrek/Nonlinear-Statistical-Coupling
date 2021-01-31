# -*- coding: utf-8 -*-
import math
import numpy as np
from typing import List
from .coupled_normal import CoupledNormal
from ..util.function import coupled_exponential


class MultivariateCoupledNormal(CoupledNormal):
    """Multivariate Coupled Normal Distribution.

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
        if validate_args:
            assert isinstance(loc, (list, np.ndarray)), "loc must be either a list or ndarray type. Otherwise use CoupledNormal."
            assert isinstance(scale, (list, np.ndarray)), "scale must be either a list or ndarray type. Otherwise use CoupledNormal."
        super(MultivariateCoupledNormal, self).__init__(
            loc=loc,
            scale=scale,
            kappa=kappa,
            alpha=alpha,
            validate_args=validate_args
        )
        if self._rank(self.scale) == 1:
            self.scale = np.diag(self.scale)
        # Ensure that scale is indeed positive definite
        # assert self.is_positive_definite(self.scale), "scale must be positive definite."
        
    # Credit: https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    def is_positive_definite(self, A: np.ndarray) -> bool:
        if np.array_equal(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def _batch_shape(self) -> List:
        if self._rank(self.loc) == 1:
            # return [] signifying single batch of a single distribution
            return []
        else:
            # return [batch size]
            return list(self.loc.shape[:-1])

    def _event_shape(self) -> List:
        if self._rank(self.loc) == 1:
            # if loc is only a vector
            return list(self.loc.shape)
        else:
            # return [n of random variables] when rank >= 2
            return [self.loc.shape[-1]]

    def _rank(self, value: [int, float, np.ndarray]) -> int:
        # specify the rank of a given value, with rank=0 for a scalar and rank=ndim for an ndarray
        if isinstance(value, (int, float)):
            return 0 
        else:
            return len(value.shape)

    def prob(self, X: [List, np.ndarray]) -> np.ndarray:
        assert X.shape[-1] ==  self.loc.shape[-1], "input X and loc must have the same dims."
        sigma = np.matmul(self.scale, self.scale)
        sigma_inv = np.linalg.inv(sigma)
        _normalized_X = lambda x: np.linalg.multi_dot([x, sigma_inv, x])
        X_norm = np.apply_along_axis(_normalized_X, 1, X)
        norm_term = self._normalized_term()
        p = (coupled_exponential(X_norm, self.kappa))**(-1/self.alpha) / norm_term
        return p

    # Normalization of the multivariate Coupled Gaussian (NormMultiCoupled)
    def _normalized_term(self) -> [int, float, np.ndarray]:
        sigma = np.matmul(self.scale, self.scale.T)
        sigma_det = np.linalg.det(sigma)
        if self.alpha == 1:
            return sigma_det**0.5 / (1 + (-1 + self.dim)*self.kappa)
        else:  # self.alpha == 2
            gamma_num = math.gamma((1 + (-1 + self.dim)*self.kappa) / (2*self.kappa))
            gamma_dem = math.gamma((1 + self.dim*self.kappa) / (2*self.kappa))
            return (np.sqrt(np.pi) * sigma_det**0.5 * gamma_num) / (np.sqrt(self.kappa) * gamma_dem)