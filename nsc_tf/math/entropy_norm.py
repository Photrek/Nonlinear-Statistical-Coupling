# -*- coding: utf-8 -*-
from math import pi

from tensorflow import repeat
from tensorflow.math import lgamma, exp
from tensorflow.linalg import det, diag_part
from .entropy import importance_sampling_integrator
from .function import coupled_logarithm
from ..distributions.multivariate_coupled_normal import MultivariateCoupledNormal


def coupled_normal_entropy(sigma, kappa):
    """
    This function calculates the coupled entropy of a coupled Gaussian 
    distribution using its sigma matrix and kappa value.
    
    Inputs
    -------
    sigma : tf.Tensor
        The equivalent of a scale matrix for a coupled Gaussian distribution.
    kappa : float
        A positive coupling value.
        
    Returns
    -------
    entropy : float
        The coupled entropy of the coupled Gaussian distribution with the 
        covariance matrix equivalent of sigma and coupling value kappa.
    """
    
    # Make sure that sigma is either 2-D or 3-D.
    assert ((len(sigma.shape) == 2) 
            | (len(sigma.shape) == 3)), ("sigma must be a 2-D tensor or a",
                                        "3-D tensor.")
                                         
    # If sigma is 2-D, the two dimensions should match so the matrix is square.
    if (len(sigma.shape) == 2):
        assert sigma.shape[0] == sigma.shape[1], ("The scale matrix must ",
                                                  "have the same number of ",
                                                  "dimensions on each side.")
        # Find the number of dimensions using the square matrix sigma.
        dim = sigma.shape[0]
    
    # If sigma is 3-D, the last two dimensions should be of the same size.
    else:
        assert sigma.shape[1] == sigma.shape[2], ("The scale matrices must ",
                                                  "have the same number of ",
                                                  "dimensions on each side.")
        # Find the number of dimensions using the square matrices in sigma.
        dim = sigma.shape[1]
        
    # If the distribution is 1-D, the determinant is just the single value in
    # sigma.
    if dim == 1:
        determinant = sigma[tuple(repeat(0, len(sigma.shape)))]
    # Otherwise, calculate the determinant of the sigma matrix.
    else:
        determinant = det(sigma)
    
    # Create the gamma function using the log-gamma() and exp() functions.
    gamma = lambda x: exp(lgamma(x))
    
    # The coupled entropy calculation is broken up over several lines.
    entropy = (((pi/kappa)**dim) * determinant)**(kappa/(1+dim*kappa))
    entropy *= (1+dim*kappa)
    entropy *= (gamma(1/(2*kappa))/gamma(0.5*(dim + 1/kappa)))**(2*kappa
                                                                /(1+dim*kappa))
    entropy += -1
    entropy /= (2*kappa)
    
    # Return the coupled entropy.
    return entropy



def biased_coupled_probability_norm(coupled_normal, kappa, alpha):
    """
    

    Parameters
    ----------
    coupled_normal : TYPE
        DESCRIPTION.
    kappa : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.

    Returns
    -------
    new_dist : TYPE
        DESCRIPTION.

    """
    dim = coupled_normal.dim
    
    scale_mult = ((1 + dim*kappa)
                  /(1 + kappa*(dim + alpha 
                               + dim*alpha*coupled_normal.kappa)))**(1/alpha)
    
    new_kappa = ((coupled_normal.kappa + dim*kappa*coupled_normal.kappa)
                 /(1 + kappa*(dim + alpha + dim*alpha*coupled_normal.kappa)))
    
    new_dist = MultivariateCoupledNormal(loc=coupled_normal.loc, 
                                         scale=diag_part(coupled_normal.scale 
                                                       * scale_mult), 
                                         kappa=new_kappa)
    return new_dist



def coupled_probability_norm(coupled_normal,
                             kappa = 0.0, 
                             alpha = 2.0):
    """
    

    Parameters
    ----------
    coupled_normal : TYPE
        DESCRIPTION.
    kappa : TYPE, optional
        DESCRIPTION. The default is 0.0.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Return the new functions that calculates the coupled density of a value.
    return biased_coupled_probability_norm(coupled_normal, kappa, alpha).prob


def coupled_cross_entropy_norm(dist_p,
                               dist_q,
                               kappa: float = 0.0, 
                               alpha: float = 2.0, 
                               root: bool = False,
                               n=10000,
                               seed=1
                               ):
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 2.0.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Fit a coupled_probability function to density_func_p with the other
    # given parameters.
    my_coupled_probability = coupled_probability_norm(dist_p,
                                                      kappa = kappa, 
                                                      alpha = alpha
                                                      )
    
    dim = dist_p.dim
    
    def raised_density_func_q(x):
        return dist_q.prob(x)**(-alpha)
    
    if root == False:
        
        def no_root_coupled_cross_entropy(x):
            
            return (my_coupled_probability(x) * \
                    (1/-alpha) * \
                    coupled_logarithm(value=raised_density_func_q(x),
                                      kappa=kappa, 
                                      dim=dim
                                      )
                    )
        
        # Integrate the function.
        final_integration = -importance_sampling_integrator(no_root_coupled_cross_entropy, 
                                                            pdf=dist_p.prob,
                                                            sampler=dist_p.sample_n, 
                                                            n=n,
                                                            seed=seed
                                                            )
        
    else:
        print("Not implemented yet.")
        pass
        
    return final_integration


def coupled_entropy_norm(dist,
                         kappa: float = 0.0, 
                         alpha: float = 2.0, 
                         root: bool = False,
                         n=10000,
                         seed=1
                         ):
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    return coupled_cross_entropy_norm(dist,
                                 dist,
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 root=root,
                                 n=n,
                                 seed=seed)


def coupled_kl_divergence_norm(dist_p, 
                               dist_q, 
                               kappa: float = 0.0, 
                               alpha: float = 2.0, 
                               root: bool = False,
                               n=10000,
                               seed=1
                               ):
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    kappa : float, optional
        DESCRIPTION. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    root : bool, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is 10000.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """    
    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy_norm(dist_p,
                                                                dist_q,
                                                                kappa=kappa,
                                                                alpha=alpha,
                                                                root=root,
                                                                n=n,
                                                                seed=seed)
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy_norm(dist_p, 
                                                     kappa=kappa, 
                                                     alpha=alpha, 
                                                     root=root,
                                                     n=n,
                                                     seed=seed)
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p