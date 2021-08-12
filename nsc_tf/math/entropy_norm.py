# -*- coding: utf-8 -*-
from math import pi

from tensorflow import repeat, squeeze, reduce_mean, matmul, expand_dims, transpose
from tensorflow.random import set_seed
from tensorflow.math import log, lgamma, exp
from tensorflow.linalg import det, diag_part, inv, trace
from .function import coupled_logarithm
from ..distributions.multivariate_coupled_normal import MultivariateCoupledNormal

def importance_sampling_integrator_norm(function, pdf, sampler, n=1000, seed=1):
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
      The function being integrated. It can have multiple outputs if they
      are batched.
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
    
    # Set a random seed for reproducibility.
    set_seed(seed)
    
    # Generate n random samples using the sampling function.
    samples = sampler(n)
    # Evaluate the function being integrated on the samples.
    weighted_function_samples = squeeze(function(samples), axis=[-2, -1])
    # Weight the samples by their associated probability densities.
    weighted_function_samples /= squeeze(pdf(samples), axis=[-2, -1])
    
    # Return the estimated integral values for the function outputs.
    return reduce_mean(weighted_function_samples, axis=1)

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
    my_coupled_probability = biased_coupled_probability_norm(
        dist_p,
        kappa=kappa,
        alpha=alpha
        ).prob
    
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
        final_integration = -importance_sampling_integrator_norm(
            no_root_coupled_cross_entropy,
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


def kl_divergence(dist_p, dist_q):
    """This function calculates the KL divergence between two Multivariate
    Gaussian distributions.

    Parameters
    ----------
    dist_p : MultivariateCoupledNormal
        The distribution whose relative entropy is being measured.
    dist_q : MultivariateCoupledNormal
        The reference distribution.

    Returns
    -------
    kl_div : tf.Tensor
        The analytical KL divergence between two multivariate Gaussians.
    """

    # Store the locs for easy access.
    loc_p = dist_p.loc
    loc_q = dist_q.loc

    # Use the scales (std. devs.) to get the covariance matrices.
    cov_p = matmul(dist_p.scale, dist_p.scale, adjoint_b=True)
    cov_q = matmul(dist_q.scale, dist_q.scale, adjoint_b=True)

    # Find the difference between the locs and insert a dimension at the 
    # beginning for broadcasting.
    loc_diff = expand_dims(loc_q - loc_p, axis=1)

    # Create a list from the shape of the loc_diff without the batch 
    # dimension.
    loc_diff_dims = [i for i in range(1, len(loc_diff.shape))]
    # Reverse the non-batch dimension shapes.
    loc_diff_dims.reverse()
    # Create the shape of the transpose of the loc_diff for each batch.
    loc_diff_dims_transpose = [0] + loc_diff_dims

    # Calculate the KL Divergence
    kl_div = log(det(cov_q)) - log(det(cov_p)) - loc_p.shape[1]
    kl_div += trace(matmul(inv(cov_q), cov_p))
    kl_div += squeeze(
        matmul(
            matmul(loc_diff, inv(cov_q)), 
            transpose(loc_diff, loc_diff_dims_transpose)
            )
        )
    kl_div *= 0.5

    return kl_div


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

    # If the Coupled Gaussians have 0 coupling and the user wants the KL
    # Divergence, return the KL divergence of the two multivariate Gaussians.
    if (dist_p.kappa, dist_q.kappa, kappa) == (0., 0., 0.):
      # Calculate the KL divergence of the two multivariate Gaussians.
      divergence =  kl_divergence(dist_p, dist_q)

    # Otherwise return the approximate coupled divergence.
    else:
      # Calculate the coupled cross-entropy of the dist_p and dist_q.
      coupled_cross_entropy_of_dists = coupled_cross_entropy_norm(
          dist_p,
          dist_q,
          kappa=kappa,
          alpha=alpha,
          root=root,
          n=n,
          seed=seed
          )
      # Calculate the  coupled entropy of dist_p
      coupled_entropy_of_dist_p = coupled_entropy_norm(
          dist_p,
          kappa=kappa,
          alpha=alpha,
          root=root,
          n=n,
          seed=seed
          )
      # Calculate the approximate coupled divergence.    
      divergence = coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p

    # Return the calculated divergence.
    return divergence