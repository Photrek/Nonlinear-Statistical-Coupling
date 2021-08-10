# -*- coding: utf-8 -*-
from tf import reduce_mean
from tf.random import set_seed
import numpy as np
from .function import coupled_logarithm, coupled_exponential


def importance_sampling_integrator(function, pdf, sampler, n=10000, seed=1):
    """
    This function performs Monte Carlo integration using importance sampling.
    It takes in a function to be integrated, the probability density of a
    distribution used to generate random numbers of the same domain as the
    function being integrated, a sampling function for that distribution, the
    number of random samples to use, and a random seed. It returns a tensor
    of the estimate(s) for the integral of the function.

    Parameters
    ----------
    function : function
        The function being integrated.
    pdf : function
        The probability density function of the distribution generating the 
        random numbers.
    sampler : function
        A function that takes in a parameter, n, and returns n random numbers.
    n : TYPE, optional
        Number of random numbers to generate for the estimates. 
        The default is 10000.
    seed : int or float, optional
        A random seed for reproducibility. The default is 1.

    Returns
    -------
    tf.Tensor
        The estimated integral(s) of the function over the support of the
        sampling distribution.

    """
    # Set a random seed.
    set_seed(seed)
    
    # Generate n samples from the probability distribution.
    samples = sampler(n)
    # Evaluate the function at the samples and divide by the probability 
    # density of the distribution at those samples.
    sampled_values = function(samples) / pdf(samples)
    
    # Return the mean sampled values as the estimates.
    return reduce_mean(sampled_values, axis=0)


def coupled_probability(density_func,
                        sampler,
                        kappa = 0.0, 
                        alpha = 1.0, 
                        dim = 1,
                        n = 10000,
                        rounds=1,
                        seed=1):
    """
    

    Parameters
    ----------
    density_func : TYPE
        DESCRIPTION.
    sampler : TYPE
        DESCRIPTION.
    kappa : TYPE, optional
        DESCRIPTION. The default is 0.0.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.0.
    dim : TYPE, optional
        DESCRIPTION. The default is 1.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    rounds : TYPE, optional
        DESCRIPTION. The default is 5.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    
    # Calculate the risk-bias.
    kMult = (-alpha * kappa) / (1 + dim*kappa)
    
    def raised_density_func(x):
        return density_func(x) ** (1-kMult)
    

    def raised_density_func_integration(x):
        return density_func(x) ** (1-kMult)
    
    # Calculate the normalization factor to the coupled CDF equals 1.
    division_factor = importance_sampling_integrator(raised_density_func_integration, 
                                                     pdf=density_func,
                                                     sampler=sampler, 
                                                     n=n,
                                                     rounds=rounds,
                                                     seed=seed)
    
    
    # Define a function to calculate coupled densities
    def coupled_prob(values):
        return raised_density_func(values) / division_factor
    
    # Return the new functions that calculates the coupled density of a value.
    return coupled_prob


def coupled_cross_entropy(density_func_p, 
                          density_func_q, 
                          sampler_p,
                          kappa: float = 0.0, 
                          alpha: float = 1.0, 
                          dim: int = 1,
                          root: bool = False,
                          n=10000,
                          rounds=1,
                          seed=1) -> [float, np.ndarray]:
    """
    

    Parameters
    ----------
    density_func_p : TYPE
        DESCRIPTION.
    density_func_q : TYPE
        DESCRIPTION.
    sampler_p : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        DESCRIPTION. The default is 1.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    rounds : TYPE, optional
        DESCRIPTION. The default is 5.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    [float, np.ndarray]
        DESCRIPTION.

    """
    
    # Fit a coupled_probability function to density_func_p with the other
    # given parameters.
    my_coupled_probability = coupled_probability(density_func=density_func_p,
                                                 sampler=sampler_p,
                                                 kappa=kappa, 
                                                 alpha=alpha,
                                                 dim=dim, 
                                                 n=n,
                                                 rounds=rounds,
                                                 seed=seed)
    
    def raised_density_func_q(x):
        return density_func_q(x)**(-alpha)
    
    if root == False:
        
        def no_root_coupled_cross_entropy(x):
            
            return (my_coupled_probability(x)
                    *(1/-alpha)
                    *coupled_logarithm(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim))
        
        # Integrate the function.
        final_integration = -importance_sampling_integrator(no_root_coupled_cross_entropy, 
                                                            pdf=density_func_p,
                                                            sampler=sampler_p, 
                                                            n=n,
                                                            rounds=rounds,
                                                            seed=seed)
        
    else:
        def root_coupled_cross_entropy(x):

            return (my_coupled_probability(x)
                    *coupled_logarithm(value=raised_density_func_q(x),
                                          kappa=kappa, 
                                          dim=dim)**(1/alpha))
        
        # Integrate the function.
        final_integration = importance_sampling_integrator(root_coupled_cross_entropy, 
                                                           pdf=density_func_p,
                                                           sampler=sampler_p, 
                                                           n=n,
                                                           rounds=rounds,
                                                           seed=seed)
        
    return final_integration


def coupled_entropy(density_func, 
                    sampler,
                    kappa: float = 0.0, 
                    alpha: float = 1.0, 
                    dim: int = 1, 
                    root: bool = False,
                    n=10000,
                    rounds=1,
                    seed=1) -> [float, np.ndarray]:
    """
    

    Parameters
    ----------
    density_func : TYPE
        DESCRIPTION.
    sampler : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        DESCRIPTION. The default is 1.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    rounds : TYPE, optional
        DESCRIPTION. The default is 1.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    [float, np.ndarray]
        DESCRIPTION.

    """

    
    return coupled_cross_entropy(density_func, 
                                 density_func, 
                                 sampler_p=sampler,
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 dim=dim,
                                 root=root,
                                 n=n,
                                 rounds=rounds,
                                 seed=seed
                                 )


def coupled_kl_divergence(density_func_p, 
                       density_func_q, 
                       sampler_p,
                       kappa: float = 0.0, 
                       alpha: float = 1.0, 
                       dim: int = 1, 
                       root: bool = False,
                       n=10000,
                       rounds=1,
                       seed=1) -> [float, np.ndarray]:
    """
    

    Parameters
    ----------
    density_func_p : TYPE
        DESCRIPTION.
    density_func_q : TYPE
        DESCRIPTION.
    sampler_p : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        DESCRIPTION. The default is 1.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    rounds : TYPE, optional
        DESCRIPTION. The default is 1.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    [float, np.ndarray]
        DESCRIPTION.

    """
    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy(density_func_p,
                                                           density_func_q,
                                                           sampler_p=sampler_p,
                                                           kappa=kappa,
                                                           alpha=alpha, 
                                                           dim=dim,
                                                           root=root,
                                                           n=n,
                                                           rounds=rounds,
                                                           seed=seed
                                                           )
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy(density_func_p, 
                                                sampler=sampler_p,
                                                kappa=kappa, 
                                                alpha=alpha, 
                                                dim=dim,
                                                root=root,
                                                n=n,
                                                rounds=rounds,
                                                seed=seed
                                                )
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p


def generalized_mean(values: np.ndarray, r: float = 1.0, weights: np.ndarray = None) -> float:
    """
    This function calculates the generalized mean of a 1-D array of non- 
    negative real numbers using the coupled logarithm and exponential functions.
    
    Parameters
    ----------
    values : np.ndarray
        DESCRIPTION : A 1-D numpy array (row vector) of non-negative numbers
         for which we are calculating the generalized mean.
    r : float, optional
        DESCRIPTION : The risk bias and the power of the generalized mean. 
        The default is 1.0 (Arithmetric Mean).
    weights : np.ndarray, optional
        DESCRIPTION : A 1-D numpy array of the weights for each value. 
        The default is None, which triggers a conditional to use equal weights.

    Returns gen_mean
    -------
    float
        DESCRIPTION : The coupled generalized mean.
    """
    
    assert type(values) == np.ndarray, "values must be a 1-D numpy ndarray."
    if len(values.shape) != 1:
        assert ((len(values.shape) == 2) 
                & ((values.shape[0] == 1)
                  | (values.shape[1] == 1))), "values must be a 1-D numpy ndarray."
    assert (values <= 0).sum() == 0, "all numbers in values must be greater than 0."
    assert ((type(r) == int) | (type(r) == float) | (type(r) == np.int32 ) 
            | (type(r) == np.float32) | (type(r) == np.int64) 
            | (type(r) == np.float64)), "r must be a numeric data type, like a float or int."
    assert ((type(weights) == type(None))
            | (type(weights) == np.ndarray)), "weights must either be None or 1-D numpy ndarray."
            
    # If weights equals None, equally weight all observations.
    if type(weights) == type(None):
        weights = weights or np.ones(len(values))
    
    # Calculate the log of the generalized mean by taking the dot product of the
    # weights vector and the vector of the coupled logarithm of the values and
    # divide the result by the sum of the the weights.
    log_gen_mean = np.dot(weights, coupled_logarithm(values, kappa=r, dim=0)) / np.sum(weights)
        
    # Calculate the generalized mean by exponentiating the log-generalized mean.
    gen_mean = coupled_exponential(log_gen_mean, kappa=r, dim=0)
    
    # Return the generalized mean.
    return gen_mean
