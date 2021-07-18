# -*- coding: utf-8 -*-
import numpy as np
#from .function import coupled_logarithm, coupled_exponential
from nsc import log as coupled_logarithm
import typing

def coupled_cross_entropy_prob(probs_p, 
                               probs_q, 
                               kappa: float = 0.0,
                               dim=1
                               ) -> [float, np.ndarray]:
    """
    This function calculates the coupled cross-entropy between the 
    probabilities (or probability densities) from two discrete (or continuous)
    distributions evaluated on the same underlying events.

    Parameters
    ----------
    probs_p : numpy.ndarray or tensorflow.Tensor
        Probabilities of x from distribution p.
    probs_q : numpy.ndarray or tensorflow.Tensor
        Probabilities of x from distribution q.
    kappa : float, optional
        Degree of coupling used in the coupled logarithm. The default is 0.0.
    dim : int, optional
        Number of dimensions a of the random variable. The default is 1, 
        assuming a univariate distribution.

    Returns
    -------
    float
        The coupled cross-entropy between p and q.

    """

    # Take the coupled logarithm of the q(x).
    log_q = coupled_logarithm(probs_q, kappa=kappa, dim=dim)
    
    # If log_q is a vector, transpose it a
    if len(log_q.shape) == 1:
        log_q_t = np.transpose(log_q)
    else:
                
        probs_p = np.expand_dims(probs_p, axis=-1)
        log_q = np.expand_dims(log_q, axis=-1)
        
        range_of_dims = [i for i in range(1, len(log_q.shape))]
        range_of_dims.reverse()
        range_of_dims = [0] + range_of_dims
        
        log_q_t = np.transpose(log_q, axes=range_of_dims)

    # Calculate the negative sum of p(x) * coupled-log(q(x), kappa).
    return -np.matmul(log_q_t, probs_p)


def coupled_entropy_prob(probs_p, 
                         kappa: float = 0.0
                         ) -> [float, np.ndarray]:
    """
    This function calculates the coupled entropy of the probabilities 
    (or probability densities) from a discrete (or continuous)
    distribution.

    Parameters
    ----------
    probs_p : numpy.ndarray or tensorflow.Tensor
        Probabilities of x from distribution p.
    kappa : float, optional
        Degree of coupling used in the coupled logarithm. The default is 0.0.

    Returns
    -------
    float
        The coupled entropy of p.

    """

    # Return the cross-entropy of p with itself, which is the entropy of p.
    return coupled_cross_entropy_prob(probs_p, 
                                      probs_p, 
                                      kappa=kappa
                                      )


def coupled_kl_divergence_prob(probs_p, 
                               probs_q, 
                               kappa: float = 0.0
                               ) -> [float, np.ndarray]:
    """
    

    Parameters
    ----------
    probs_p : numpy.ndarray or tensorflow.Tensor
        Probabilities of x from distribution p.
    probs_q : numpy.ndarray or tensorflow.Tensor
        Probabilities of x from distribution q.
    kappa : float, optional
        Degree of coupling used in the coupled logarithm. The default is 0.0.

    Returns
    -------
    float
        The coupled KL divergence between p and q.

    """
    
    
    # Calculate the coupled cross-entropy of p and q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy_prob(probs_p,
                                                                probs_q,
                                                                kappa=kappa
                                                                )
    # Calculate the coupled entropy of p.
    coupled_entropy_of_dist_p = coupled_entropy_prob(probs_p,
                                                     kappa=kappa
                                                     )
    
    # Return the coupled KL divergence of p and q.
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p
