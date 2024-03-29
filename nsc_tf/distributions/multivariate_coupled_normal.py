# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import ipdb
import tensorflow_probability as tfp
from typing import List, Union
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
                 loc: [int, float, List, np.ndarray, tf.Tensor],
                 scale: [int, float, List, np.ndarray, tf.Tensor],
                 kappa: [int, float] = 0.,
                 alpha: int = 2,
                 validate_args: bool = True
                 ):
        if validate_args:
            assert isinstance(loc, (list, np.ndarray, tf.Tensor, tf.Variable)), \
                "loc must be either a list or array like. Otherwise use CoupledNormal."
            assert isinstance(scale, (list, np.ndarray, tf.Tensor, tf.Variable)), \
                "scale must be either a list or array like. Otherwise use CoupledNormal."
        super(MultivariateCoupledNormal, self).__init__(
            loc=loc,
            scale=scale,
            kappa=kappa,
            alpha=alpha,
            validate_args=validate_args
        )
        
        #Generating covariance matrices
        if self._batch_shape:
            #Tensorflow
            if isinstance(scale, (tf.Tensor, tf.Variable)):
                _scale_diag = tf.linalg.diag(scale)
                for _scale_batch in _scale_diag:
                    assert self.is_positive_definite(_scale_batch), \
                        "scale must be positive definite, but not necessarily symmetric."
                self._scale = _scale_diag
            #Else numpy
            else:
                _scale_diag = np.empty(shape=[d for d in self._scale.shape]+\
                                       [self._scale.shape[-1]]
                                       )
                # This can be further optimized
                for i, _scale_batch in enumerate(self._scale.numpy()):
                    # Ensure that scale sub-batch is indeed positive definite
                    assert self.is_positive_definite(np.diag(_scale_batch)), \
                        "scale must be positive definite, but not necessarily symmetric."
                    _scale_diag[i] = np.diag(_scale_batch)
                self._scale = _scale_diag
        else:
            self._scale = np.diag(self._scale)
            # Ensure that scale is indeed positive definite
            assert self.is_positive_definite(self._scale), \
                "scale must be positive definite, but not necessarily symmetric."
                
        # need to revisit this if self._scale contains lower triangle and not diagonal
        #Tensorflow
        if isinstance(self._scale, tf.Tensor):
            self._sigma = tf.linalg.matmul(self._scale, self._scale)
        #Numpy
        else:
            self._sigma = np.matmul(self._scale, self._scale)
        self._norm_term = self._get_normalized_term() 
            
    def _get_batch_shape(self) -> List:
        if self._rank(self._loc) == 1:
            # return [] signifying single batch of a multivariate distribution
            return []
        else:
            # return [batch size]
            return [batch_dim for batch_dim in self._loc.shape[:-1]]

    def _get_event_shape(self) -> List:
        # return the number of the multivariate dimensions.
        return [self._loc.shape[-1]]

    # Credit: https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    # This is only for positive definite, not symmetric positive definite
    def is_positive_definite(self, A: [np.ndarray, tf.Tensor]) -> bool:
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return False
        return True
        
    def _rank(self, value: [int, float, np.ndarray]) -> int:
        # specify the rank of a given value, with rank=0 for a scalar and rank=ndim for an ndarray
        if isinstance(value, (int, float)):
            return 0 
        else:
            return len(value.shape)
        
    
    def _sample_(self, loc, Scale, kappa=0.0, n=1):
        """Generate random variables of multivariate coupled normal 
        distribution.
        
        Inputs
        ------
        loc : array_like
            Mean of random variable, length determines dimension of random 
            variable
        Scale : array_like
            Square array of covariance matrix
        kappa : int or float
            degree of coupling
        n : int
            Number of observations, return random array will be (n, len(loc))
            
        Returns
        -------
        samples : ndarray, (n, len(loc))
            Each row is an independent draw of a multivariate coupled normally 
            distributed random variable
        """

        # Convert loc to an array, if it is not already.
        loc = np.asarray(loc)
        # Find the number of dimensions of the distribution from the loc array.
        dim = len(loc)

        # If kappa is 0, x is equal to 1.
        if kappa == 0.0:
            x = 1.0
        # Otherwise, draw n samples of x where x_i ~ Chi-sq(df = 1/kappa) and 
        # scale them by 1/kappa.
        else:
            x = np.random.chisquare(1.0/kappa, n) / (1.0/kappa)
            # Make sure x will broadcast correctly.
            x = x.reshape(n, 1)
        
        # Draw n samples from a multivariate normal centered at the origin 
        # with the covariance matrix equal to scale.
        z = np.random.multivariate_normal(np.zeros(dim), Scale, (n,))
        
        # Scale the z_i by the square root of x_i and add in the loc to get 
        # the random draws of the coupled normal random variables.
        samples = loc + z/np.sqrt(x)
        
        # Return the coupled normal random variables.
        return samples
    
    
    def sample_n(self, n=1):
        """Generate random variables for batches of multivariate coupled 
        normal distributions.
    
        Inputs
        ------
        n : int
            Number of observations per distribution.
            
        Returns
        -------
        samples : ndarray, (loc.shape[0], n, loc.shape[1])
            n samples from each distribution.
        """
        #ipdb.set_trace()
        loc, scale = self._loc, self._sigma
        
        # Find the number of batches.
        n_batches = self._loc.shape[0]
        
        #Tensorflow
        if isinstance(self._loc, (tf.Tensor, tf.Variable)):
            
            #Generating normal variates with covariance matrices given by scale
            mvn_dist = tfp.distributions.MultivariateNormalTriL(scale_tril=scale)
            z = mvn_dist.sample(n)
            
            #Swapping first and second axes: [n, n_batches, dim] -> [n_batches, n, dim]
            z = tf.transpose(z, perm=[1,0,2])
            
            if self._kappa == 0.0:
                x = 1.0
            else:
                chi2 = tfp.distributions.Chi2(df=1/self._kappa)
                #Chi-square denominator should have shape [n_batches, n, dim]
                x = chi2.sample(sample_shape=[n_batches, n, self._loc.shape[1]])*self._kappa
            
            #Convert loc from [n_batches, dim] -> [n_batches, 1, dim] for broadcasting
            samples = tf.expand_dims(loc, axis=1) + z/tf.math.sqrt(x)
        
        #Numpy
        else:
            # Create a list to hold the samples from each distribution.
            samples = []
            
            # Iterate through each of the batched samples' parameters.
            for i in range(n_batches):
                
                # Get the i-th loc and scale arrays.
                temp_loc = loc[i]
                temp_scale = scale[i]
                
                # Get the random draws from the i-th distribution.
                temp_samples = self._sample_(
                    temp_loc, 
                    temp_scale, 
                    kappa=self._kappa, n=n)
                # Add the samples to the list of samples.
                samples.append(temp_samples)
                
            # Convert the list of samples to a 3-D array.
            samples = np.array(samples, ndmin=3)
        
        # Return the samples.
        return samples
    
    def prob(self, X: tf.Tensor) -> tf.Tensor: # John removed beta_func as an argument because it didn't appear elsewhere.
        # assert X.shape[-1] ==  self._loc.shape[-1], "input X and loc must have the same dims."
        loc = tf.expand_dims(self._loc, axis=1) # John added this to broadcast x - loc
        
        # Invert the covariance matrices.
        _sigma_inv = tf.linalg.inv(self._sigma)
        
        _norm_term = self._norm_term
        
        if self._batch_shape:
            # Demean the samples.
            demeaned_samples = X - loc
            # Create X and X_t (John added this)
            X = tf.expand_dims(demeaned_samples, axis=-1)
            X_t = tf.expand_dims(demeaned_samples, axis=-2)
            # Add in an axis at the second position for broadcasting (John added this).
            _sigma_inv = tf.expand_dims(_sigma_inv, axis=1)
            X_norm = tf.linalg.matmul(tf.linalg.matmul(X_t, _sigma_inv), X)
            
            # We want to expand _norm_term to have the same number of 
            # dimensions as X_norm.
            
            # Count the difference in dims between X_norm and _norm_term.
            dim_diff = len(X_norm.shape) - len(_norm_term.shape)
            # Create a list of the dimensions to expand.
            expanded_dims = tuple([i+1 for i in range(dim_diff)])
            # Expand those dimensions
            for dim in expanded_dims:
                _norm_term = tf.expand_dims(_norm_term, axis=dim)
            
        else:
            _normalized_X = lambda x: np.linalg.multi_dot([x-loc,
                                                           _sigma_inv,
                                                           x-loc
                                                           ]
                                                          )
            X_norm = np.apply_along_axis(_normalized_X, 1, X)
        p = (coupled_exponential(X_norm, self._kappa, self._dim))**(-1/self._alpha) \
            / _norm_term
        return p

    # Normalization constant of the multivariate Coupled Gaussian (NormMultiCoupled)    
    def _get_normalized_term(self, beta_func = True) -> [int, float, np.ndarray]:
        
        if beta_func:
            # need to revisit this if self._scale contains lower triangle and not diagnoal matrixes
#             sigma = np.matmul(self._scale, self._scale.T)
            _sigma_det = tf.linalg.det(self._sigma)
            base_term = tf.math.sqrt((2 * np.pi)**self._dim * _sigma_det)
            return base_term*self._normalization_function()
        else:
#             sigma = np.matmul(self._scale, self._scale.T)
            _sigma_det = np.linalg.det(self._sigma)
            if self._alpha == 1:
                return _sigma_det**0.5 / (1 + (-1 + self._dim)*self._kappa)
            else:  # self._alpha == 2
                gamma_num = gamma((1 + (-1 + self._dim)*self._kappa) / (2*self._kappa))
                gamma_dem = gamma((1 + self._dim*self._kappa) / (2*self._kappa))
                return (np.sqrt(np.pi) * _sigma_det**0.5 * gamma_num) / \
                    (np.sqrt(self._kappa) * gamma_dem)

    def entropy(self, kappa: [int, float] = None, root: bool = False,
                n: int = 10000, rounds: int = 1, seed: int = 1
                ) -> [float, np.ndarray]:
        return coupled_entropy(density_func=self.prob,
                               sampler=self.sample_n,
                               kappa=kappa if kappa else self.kappa,
                               alpha=self.alpha,
                               dim=self.dim,
                               root=root,
                               n=n,
                               rounds=rounds,
                               seed=seed
                               )

    def cross_entropy(self, dist_q, kappa: [int, float] = None, root: bool = False,
                      n: int = 10000, rounds: int = 1, seed: int = 1
                      ) -> [float, np.ndarray]:
        return coupled_cross_entropy(density_func_p=self.prob,
                                     density_func_q=dist_q.prob,
                                     sampler_p=self.sample_n,
                                     kappa=kappa if kappa else self.kappa,
                                     alpha=self.alpha,
                                     dim=self.dim,
                                     root=root,
                                     n=n,
                                     rounds=rounds,
                                     seed=seed
                                     )

    def kl_divergence(self, dist_q, kappa: [int, float] = None, root: bool = False,
                      n: int = 10000, rounds: int = 1, seed: int = 1
                      ) -> [float, np.ndarray]:
        return coupled_kl_divergence(density_func_p=self.prob,
                                     density_func_q=dist_q.prob,
                                     sampler_p=self.sample_n,
                                     kappa=kappa if kappa else self.kappa,
                                     alpha=self.alpha,
                                     dim=self.dim,
                                     root=root,
                                     n=n,
                                     rounds=rounds,
                                     seed=seed
                                     )
