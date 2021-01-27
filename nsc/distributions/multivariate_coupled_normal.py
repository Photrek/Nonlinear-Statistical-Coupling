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

    def prob(self, X: [List, np.ndarray]):
        pass
        '''
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
        # normCGvalue =  1/float(norm_CG(scale, kappa))
        # coupledNormalDistributionResult = normCGvalue * (coupled_exponential(y, kappa)) ** -0.5
        return p
        '''

    # Normalization of the multivariate Coupled Gaussian (NormMultiCoupled)
    def _normalized_term(self) -> [int, float, np.ndarray]:
        pass
        '''
        if self.kappa == 0:
            norm_term = math.sqrt(2*math.pi) * self.scale
        elif self.kappa < 0:
            gamma_num = math.gamma(self.kappa-1) / (2*self.kappa)
            gamma_dem = math.gamma(1 - (1 / (2*self.kappa)))
            norm_term = (math.sqrt(math.pi)*self.scale*gamma_num) / float(math.sqrt(-1*self.kappa)*gamma_dem)
        else:
            gamma_num = math.gamma(1 / (2*self.kappa))
            gamma_dem = math.gamma((1+self.kappa)/(2*self.kappa))
            norm_term = (math.sqrt(math.pi)*self.scale*gamma_num) / float(math.sqrt(self.kappa)*gamma_dem)
        return norm_term
        '''
