# -*- coding: utf-8 -*-
import numpy as np
from typing import List
from scipy.special import gamma
from .coupled_normal import CoupledNormal
from ..math.function import coupled_exponential
from ..math.entropy import coupled_entropy, \
                           coupled_cross_entropy, \
                           coupled_kl_divergence


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
        if self._rank(self._scale) == 1:
            self._scale = np.diag(self._scale)
        # Ensure that scale is indeed positive definite
        assert self.is_positive_definite(self._scale), "scale must be positive definite, but not necessarily symmetric."

    # Credit: https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    # This is only for positive definite, not symmetric positive definite
    def is_positive_definite(self, A: np.ndarray) -> bool:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def _batch_shape(self) -> List:
        if self._rank(self._loc) == 1:
            # return [] signifying single batch of a single distribution
            return []
        else:
            # return [batch size]
            return list(self._loc.shape[:-1])

    def _event_shape(self) -> List:
        if self._rank(self._loc) == 1:
            # if loc is only a vector
            return list(self._loc.shape)
        else:
            # return [n of random variables] when rank >= 2
            return [self._loc.shape[-1]]

    def _rank(self, value: [int, float, np.ndarray]) -> int:
        # specify the rank of a given value, with rank=0 for a scalar and rank=ndim for an ndarray
        if isinstance(value, (int, float)):
            return 0 
        else:
            return len(value.shape)

    def sample_n(self, n: int) -> np.array:
        # normal_samples = np.random.normal(loc=self._loc, scale=self._scale, size=n)
        mvn_samples = np.random.multivariate_normal(mean=self._loc, cov=self._scale, size=n, check_valid='warn')
        chi2_samples = np.random.chisquare(df=1/self._kappa, size=n)
        # Transpose to allow for broadcasting the following: (n x d) / (n x 1)
        samples_T = mvn_samples.T / np.sqrt(chi2_samples*self._kappa)
        samples = samples_T.T
        return self._loc + samples
        ''' TFP Source: https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/multivariate_student_t.py#L238-L254
        
        normal_seed, chi2_seed = samplers.split_seed(seed, salt='multivariate t')

        loc = tf.broadcast_to(self._loc, self._sample_shape())
        mvn = mvn_linear_operator.MultivariateNormalLinearOperator(
            loc=tf.zeros_like(loc), scale=self._scale)
        normal_samp = mvn.sample(n, seed=normal_seed)

        df = tf.broadcast_to(self.df, self.batch_shape_tensor())
        chi2 = chi2_lib.Chi2(df=df)
        chi2_samp = chi2.sample(n, seed=chi2_seed)

        return (self._loc +
                normal_samp * tf.math.rsqrt(chi2_samp / self._df)[..., tf.newaxis])
        '''

    def prob(self, X: [List, np.ndarray], beta_func: bool = True) -> np.ndarray:
        # assert X.shape[-1] ==  self._loc.shape[-1], "input X and loc must have the same dims."
        sigma = np.matmul(self._scale, self._scale)
        sigma_inv = np.linalg.inv(sigma)
        _normalized_X = lambda x: np.linalg.multi_dot([x-self._loc, sigma_inv, x-self._loc])
        # BUG FIX: Not taking into account self._loc
        X_norm = np.apply_along_axis(_normalized_X, 1, X)
        norm_term = self._normalized_term(beta_func)
        p = (coupled_exponential(X_norm, self._kappa, self._dim))**(-1/self._alpha) / norm_term
        return p

    # Normalization constant of the multivariate Coupled Gaussian (NormMultiCoupled)
    def _normalized_term(self, beta_func) -> [int, float, np.ndarray]:
        if beta_func:
            sigma = np.matmul(self._scale, self._scale.T)
            sigma_det = np.linalg.det(sigma)
            base_term = np.sqrt((2 * np.pi)**self._dim * sigma_det)
            return base_term*self._normalization_function()
        else:
            sigma = np.matmul(self._scale, self._scale.T)
            sigma_det = np.linalg.det(sigma)
            if self._alpha == 1:
                return sigma_det**0.5 / (1 + (-1 + self._dim)*self._kappa)
            else:  # self._alpha == 2
                gamma_num = gamma((1 + (-1 + self._dim)*self._kappa) / (2*self._kappa))
                gamma_dem = gamma((1 + self._dim*self._kappa) / (2*self._kappa))
                return (np.sqrt(np.pi) * sigma_det**0.5 * gamma_num) / (np.sqrt(self._kappa) * gamma_dem)

    def entropy(self, root: bool = False, n: int = 10000, rounds: int = 1,
                seed: int = 1) -> [float, np.ndarray]:
        return coupled_entropy(density_func=self.prob,
                               sampler=self.sample_n,
                               kappa=self.kappa,
                               alpha=self.alpha,
                               dim=self.dim,
                               root=root,
                               n=n,
                               rounds=rounds,
                               seed=seed
                               )

    def cross_entropy(self, dist_q, root: bool = False, n: int = 10000,
                      rounds: int = 1, seed: int = 1) -> [float, np.ndarray]:
        return coupled_cross_entropy(density_func_p=self.prob,
                                     density_func_q=dist_q.prob,
                                     sampler_p=self.sample_n,
                                     kappa=self.kappa,
                                     alpha=self.alpha,
                                     dim=self.dim,
                                     root=root,
                                     n=n,
                                     rounds=rounds,
                                     seed=seed
                                     )

    def kl_divergence(self, dist_q, root: bool = False, n: int = 10000,
                      rounds: int = 1, seed: int = 1) -> [float, np.ndarray]:
        return coupled_kl_divergence(density_func_p=self.prob,
                                     density_func_q=dist_q.prob,
                                     sampler_p=self.sample_n,
                                     kappa=self.kappa,
                                     alpha=self.alpha,
                                     dim=self.dim,
                                     root=root,
                                     n=n,
                                     rounds=rounds,
                                     seed=seed
                                     )
