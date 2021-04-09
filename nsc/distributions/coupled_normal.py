# -*- coding: utf-8 -*-
import numpy as np
from typing import List
from scipy.special import beta, gamma
from ..math.function import coupled_exponential
# from ..math.entropy_norm import coupled_entropy_norm, \
#                                 coupled_cross_entropy_norm, \
#                                 coupled_kl_divergence_norm


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
                # assert loc.shape == scale.shape, "loc and scale must have the same dimensions (check respective .shape())."
                assert np.all((scale >= 0)), "All scale values must be greater or equal to 0."            
            else:
                assert scale >= 0, "scale must be greater or equal to 0."            
            assert isinstance(kappa, (int, float)), "kappa must be an int or float type."
            assert isinstance(alpha, int), "alpha must be an int that equals to either 1 or 2."
            assert alpha in [1, 2], "alpha must be equal to either 1 or 2."
        self._loc = loc
        self._scale = scale
        self._kappa = kappa
        self._alpha = alpha
        self._dim = self._n_dim()

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def kappa(self):
        return self._kappa

    @property
    def alpha(self):
        return self._alpha

    @property
    def dim(self):
        return self._dim

    def _n_dim(self):
        return 1 if self._event_shape() == [] else self._event_shape()[0]

    def _batch_shape(self) -> List:
        if self._rank(self._loc) == 0:
            # return [] signifying single batch of a single distribution
            return []
        else:
            # return the batch shape in list format
            return list(self._loc.shape)

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

    def sample_n(self, n: int) -> np.array:
        # https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.random.gamma.html
        normal_samples = np.random.normal(loc=self._loc, scale=self._scale, size=n)
        if self._kappa == 0:
            samples = normal_samples
        else:
            gamma_samples = np.random.gamma(shape=1/(2*self._kappa), scale=self._scale, size=n)
            samples = normal_samples * 1/np.sqrt(gamma_samples*self._kappa)
        return self._loc + (samples * self._scale)
        ''' TFP Source: https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/student_t.py#L43-L79

        normal_seed, gamma_seed = samplers.split_seed(seed, salt='student_t')
        shape = ps.concat([[n], batch_shape], 0)

        normal_sample = samplers.normal(shape, dtype=dtype, seed=normal_seed)
        df = df * tf.ones(batch_shape, dtype=dtype)
        gamma_sample = gamma_lib.random_gamma(
            [n], concentration=0.5 * df, rate=0.5, seed=gamma_seed)
        samples = normal_sample * tf.math.rsqrt(gamma_sample / df)
        return samples * scale + loc
        '''

    def prob(self, X: [List, np.ndarray]) -> np.ndarray:
        # Check whether input X is valid
        X = np.asarray(X) if isinstance(X, List) else X
        assert isinstance(X, np.ndarray), "X must be a List or np.ndarray."
        # assert type(X[0]) == type(self._loc), "X samples must be the same type as loc and scale."
        if isinstance(X[0], np.ndarray):
            assert X[0].shape == self._loc.shape, "X samples must have the same dimensions as loc and scale (check respective .shape())."
        # Calculate PDF with input X
        X_norm = (X-self._loc)**2 / self._scale**2
        norm_term = self._normalized_term()
        # p is the density vector
        p = (coupled_exponential(X_norm, self._kappa))**-0.5 / norm_term
        return p

    # Normalization constant of 1-D Coupled Gaussian (NormCG)
    def _normalized_term(self) -> [int, float, np.ndarray]:
        base_term = np.sqrt(2*np.pi) * self._scale
        norm_term = base_term*self._normalization_function()
        return norm_term

    def _normalization_function(self):
        k, d = self._kappa, self._dim
        assert -1/d < k, "kappa must be greater than -1/dim."
        if k == 0:
            return 1
        elif k > 0:
            func_term = (1 + d*k) / (2*k)**(d/2)
            beta_input_x = 1/(2*k) + 1
            beta_input_y = d/2
            gamma_input = d/2
            return func_term * beta(beta_input_x, beta_input_y)/gamma(gamma_input)
        else:  # -1 < self._kappa < 0:
            func_term = 1 / (-2*k)**(d/2)
            beta_input_x = (1 + d*k)/(-2*k) + 1
            beta_input_y = d/2
            gamma_input = d/2
            return func_term * beta(beta_input_x, beta_input_y)/gamma(gamma_input)

    '''
    def entropy(self, root: bool = False, n: int = 10000, rounds: int = 1,
                seed: int = 1) -> [float, np.ndarray]:
        return coupled_entropy_norm(dist=self,
                                    kappa=self.kappa,
                                    alpha=self.alpha,
                                    root=root,
                                    n=n,
                                    rounds=rounds,
                                    seed=seed
                                    )

    def cross_entropy(self, dist_q, root: bool = False, n: int = 10000,
                      rounds: int = 1, seed: int = 1) -> [float, np.ndarray]:
        return coupled_cross_entropy_norm(dist_p=self,
                                          dist_q=dist_q,
                                          kappa=self.kappa,
                                          alpha=self.alpha,
                                          root=root,
                                          n=n,
                                          rounds=rounds,
                                          seed=seed
                                          )

    def kl_divergence(self, dist_q, root: bool = False, n: int = 10000,
                      rounds: int = 1, seed: int = 1) -> [float, np.ndarray]:
        return coupled_kl_divergence_norm(dist_p=self,
                                          dist_q=dist_q,
                                          kappa=self.kappa,
                                          alpha=self.alpha,
                                          root=root,
                                          n=n,
                                          rounds=rounds,
                                          seed=seed
                                          )
    '''

    def __repr__(self) -> str:
        return f"<nsc.distributions.{self.__class__.__name__} batch_shape={str(self._batch_shape())} event_shape={str(self._event_shape())}>"
