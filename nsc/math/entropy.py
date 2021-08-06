# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable
from .function import coupled_logarithm


def importance_sampling_integrator(function: Callable[..., np.ndarray],
                                   pdf: Callable[..., np.ndarray],
                                   sampler: Callable[..., int],
                                   n: int = 10000,
                                   seed: int = 1
                                   ) -> np.array:
    """
    

    Parameters
    ----------
    function : TYPE
        DESCRIPTION.
    pdf : TYPE
        DESCRIPTION.
    sampler : TYPE
        DESCRIPTION.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # Set a random seed.
    np.random.seed(seed)
    
    # Generate n samples from the probability distribution.
    samples = sampler(n)
    #ipdb.set_trace()
    # Evaluate the function at the samples and divide by the probability 
    # density of the distribution at those samples.
    sampled_values = function(samples) / pdf(samples)
    # Add the estimate of the integral to the estimates list.
    estimates = np.mean(sampled_values, axis=1) # Altered this for the batching.
    
    # Return the mean of the estimates as the estimate of the integral.
    return np.array(estimates)


def coupled_probability(density_func: Callable[..., np.ndarray],
                        sampler: Callable[..., int],
                        kappa: float = 0.0, 
                        alpha: float = 1.0, 
                        dim: int = 1,
                        n: int = 10000,
                        rounds: int = 1,
                        seed: int = 1
                        ) -> [float, np.ndarray]:
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
                                                     seed=seed
                                                     )
    
    
    # Define a function to calculate coupled densities
    def coupled_prob(values):
        return raised_density_func(values) / division_factor
    
    # Return the new functions that calculates the coupled density of a value.
    return coupled_prob


def coupled_cross_entropy(density_func_p: Callable[..., np.ndarray],
                          density_func_q: Callable[..., np.ndarray],
                          sampler_p: Callable[..., int],
                          kappa: float = 0.0, 
                          alpha: float = 1.0, 
                          dim: int = 1,
                          root: bool = False,
                          n: int = 10000,
                          rounds: int = 1,
                          seed: int = 1
                          ) -> [float, np.ndarray]:
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


def coupled_entropy(density_func: Callable[..., np.ndarray],
                    sampler: Callable[..., int],
                    kappa: float = 0.0,
                    alpha: float = 1.0,
                    dim: int = 1,
                    root: bool = False,
                    n: int = 10000,
                    rounds: int = 1,
                    seed: int = 1
                    ) -> [float, np.ndarray]:
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


def coupled_kl_divergence(density_func_p: Callable[..., np.ndarray],
                          density_func_q: Callable[..., np.ndarray],
                          sampler_p: Callable[..., int],
                          kappa: float = 0.0,
                          alpha: float = 1.0,
                          dim: int = 1,
                          root: bool = False,
                          n: int = 10000,
                          rounds: int = 1,
                          seed: int = 1
                          ) -> [float, np.ndarray]:
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
