# -*- coding: utf-8 -*-
import numpy as np
import math
from typing import Any  # for NDArray types


def coupled_logarithm(value: [int, float, np.ndarray], kappa: [int, float] = 0.0, dim: int = 1) -> [float, np.ndarray]:
    """
    Generalization of the logarithm function, which defines smooth
    transition to power functions.
    
    Parameters
    ----------
    value : Input variable in which the coupled logarithm is applied to.
            Accepts int, float, and np.ndarray data types.
    kappa : Coupling parameter which modifies the coupled logarithm function.
            Accepts int and float data types.
    dim : The dimension (or rank) of value. If value is scalar, then dim = 1.
          Accepts only int data type.
    """
    # convert value into np.ndarray (if scalar) to keep consistency
    value = np.array(value) if isinstance(value, (int, float)) else value
    assert isinstance(value, np.ndarray), "value must be an int, float, or np.ndarray."
    assert 0. not in value, "value must not be or contain any zero(s)."
    if kappa == 0.:
        coupled_log_value = np.log(value)  # divide by 0 if x == 0
    else:
        coupled_log_value = (1. / kappa) * (value**(kappa / (1. + dim*kappa)) - 1.)
    return coupled_log_value


def coupled_exponential(value: [int, float, np.ndarray], kappa: float = 0.0, dim: int = 1) -> [float, np.ndarray]:
    """
    Generalization of the exponential function.
    
    Parameters
    ----------
    value : [float, Any]
        Input values in which the coupled exponential is applied to.
    kappa : float,
        Coupling parameter which modifies the coupled exponential function. 
        The default is 0.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.
    
    Returns
    -------
    float
        The coupled exponential values.
    
    """
    # convert number into np.ndarray to keep consistency
    value = np.array(value) if isinstance(value, (int, float)) else value
    assert isinstance(value, np.ndarray), "value must be an int, float, or np.ndarray."
    assert 0 not in value, "value must not be or contain any zero(s)."
    assert isinstance(dim, int) and dim >= 0, "dim must be an integer greater than or equal to 0."
    # check that -1/d <= kappa
    assert -1/dim <= kappa, "kappa must be greater than or equal to -1/dim."

    if kappa == 0:
        coupled_exp_value = np.exp(value)
    elif kappa > 0:
        # coupled_exp_value = (1 + kappa*value)**((1 + dim*kappa)/kappa)
        coupled_exp_value = (1 + kappa*value)**(1 / (kappa / (1 + dim*kappa)))
    # the following is given that kappa < 0
    else:
        def _compact_support(value, kappa, dim):
            if (1 + kappa*value) >= 0:
                try:
                    # return (1 + kappa*value)**((1 + dim*kappa)/kappa)
                    return (1 + kappa*value)**(1 / (kappa / (1 + dim*kappa)))
                except ZeroDivisionError:
                    print("Skipped ZeroDivisionError at the following: " + \
                          f"value = {value}, kappa = {kappa}. Therefore," + \
                          f"(1+kappa*value) = {(1+kappa*value)}"
                          )
            elif ((1 + dim*kappa)/kappa) > 0:
                return 0.
            else:
                return float('inf')    
        compact_support = np.vectorize(_compact_support)
        coupled_exp_value = compact_support(value, kappa, dim)

    return coupled_exp_value


def coupled_probability(dist, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    [float, Any]
        DESCRIPTION.

    """
    
    # Calculate the risk-bias.
    kMult = (-alpha * kappa) / (1 + dim*kappa)
    # Raise the distribution densities to 1 - the risk-bias
    new_dist_temp = dist ** (1-kMult)
    # Forget dist inside the function to free up memory.
    del dist
    # Calculate the normalization factor to the coupled CDF equals 1.
    division_factor = np.trapz(new_dist_temp, dx=dx)
    # Calculate the coupled densities
    coupled_dist = new_dist_temp / division_factor

    return coupled_dist


def coupled_cross_entropy(dist_p, dist_q, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1, root: bool = False) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.
    root : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    [float, Any]
        DESCRIPTION.

    """

    if root == False:
        # Raise the distrubtion P to the power (-alpha*kappa)/(1+dim*kapaa) 
        # and normalize it. 
        dist_p_temp = coupled_probability(dist=dist_p, 
                                          dx=dx, 
                                          kappa=kappa, 
                                          alpha=alpha, 
                                          dim=dim)
        # Forget dist_p inside the fuction to save memory.
        del dist_p
        
        # Calculate the coupled-logarithm of the values in the distribution Q 
        # raised to the negative alpha power.
        coupled_logarithm_dist_q = (1/-alpha)*coupled_logarithm(value=dist_q**(-alpha), 
                                                                kappa=kappa, 
                                                                dim=dim)
        # Forget dist_q inside the fuction to save memory.
        del dist_q
        
        # Multiply the coupled-probability values of dist_p by 
        # (1/-alpha)*coupled logarithm of dist_q.
        pre_integration = np.multiply(dist_p_temp, 
                                      coupled_logarithm_dist_q)
        # Integrate the values and multiply by negative one.
        final_integration = -np.trapz(pre_integration, dx=dx)
        
    else:
        # Raise the distrubtion P to the power (-alpha*kappa)/(1+dim*kapaa) 
        # and normalize it. 
        dist_p_temp = coupled_probability(dist=dist_p, 
                                          dx=dx, 
                                          kappa=kappa, 
                                          alpha=alpha, 
                                          dim=dim)
        # Forget dist_p inside the fuction to save memory.
        del dist_p
        # Calculate the coupled logarithm of distribution Q raised to the
        # negative alpha power and raise those values to the (1/alpha) power.
        coupled_logarithm_dist_q = coupled_logarithm(value=dist_q**(-alpha), 
                                                     kappa=kappa, 
                                                     dim=dim)**(1/alpha)
        # Forget dist_q inside the fuction to save memory.
        del dist_q
        

        # Multiply the coupled-probability values of dist_p by coupled 
        # logarithm of dist_q.
        pre_integration = np.multiply(dist_p_temp, 
                                      coupled_logarithm_dist_q)
        # Integrate the values.
        final_integration = np.trapz(pre_integration, dx=dx)
        
    return final_integration


def coupled_entropy(dist, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1, root: bool = False) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.
    root : bool, optional
        DESCRIPTION. The default is false.

    Returns
    -------
    [float, Any]
        The coupled cross-entropy between dist_p and dist_q.

    """
    
    return coupled_cross_entropy(dist_p=dist, 
                                 dist_q=dist, 
                                 dx=dx,
                                 kappa=kappa, 
                                 alpha=alpha, 
                                 dim=dim,
                                 root=root)


def coupled_divergence(dist_p, dist_q, dx: float, kappa: float = 0.0, alpha: float = 1.0, dim: int = 1, root: bool = False) -> [float, Any]:
    """
    

    Parameters
    ----------
    dist_p : TYPE
        DESCRIPTION.
    dist_q : TYPE
        DESCRIPTION.
    dx : float
        The distance between realizations of the densities.
    kappa : float, optional
        Coupling parameter. The default is 0.0.
    alpha : float, optional
        DESCRIPTION. The default is 1.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.
    root : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    [float, Any]
       The coupled divergence.

    """
    
    # Calculate the coupled cross-entropy of the dist_p and dist_q.
    coupled_cross_entropy_of_dists = coupled_cross_entropy(dist_p=dist_p, 
                                                           dist_q=dist_q, 
                                                           dx=dx,
                                                           kappa=kappa, 
                                                           alpha=alpha, 
                                                           dim=dim,
                                                           root=root)
    # Calculate the  coupled entropy of dist_p
    coupled_entropy_of_dist_p = coupled_entropy(dist=dist_p, 
                                                dx=dx, 
                                                kappa=kappa, 
                                                alpha=alpha, 
                                                dim=dim,
                                                root=root)
    
    return coupled_cross_entropy_of_dists - coupled_entropy_of_dist_p


def tsallis_entropy(dist, kappa, dx, alpha = 1, dim = 1, normalize = False, root = False):
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    kappa : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    dim : TYPE, optional
        DESCRIPTION. The default is 1.
    normalize : bool, optional
        DESCRIPTION. The default is False.
    root : False, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    if normalize:
        entropy = (1+kappa)**(1/alpha) * coupled_entropy(dist=dist, 
                                                              dx=dx, 
                                                              kappa=kappa, 
                                                              alpha=alpha, 
                                                              dim=dim, 
                                                              root=root)
    else:
        entropy = (np.trapz(dist**(1+(alpha*kappa/(1+kappa))), dx=dx) 
                        * (1+kappa)**(1/alpha)
                        * coupled_entropy(dist=dist,
                                          dx=dx, 
                                          kappa=kappa,
                                          alpha=alpha,
                                          dim=dim,
                                          root=root))
    
    return entropy


def shannon_entropy(dist, dx, dim = 1, root = False):
    """
    

    Parameters
    ----------
    dist : TYPE
        DESCRIPTION.
    dx : float
        DESCRIPTION.
    dim : int, optional
        DESCRIPTION. The default is 1.
    root : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if root:
        alpha = 2
    else: 
        alpha = 1
    
    return coupled_entropy(dist, 
                           dx=dx,   
                           kappa=0.0, 
                           alpha=alpha, 
                           dim=dim, 
                           root=root)


def norm_CG(sigma, kappa):
    if kappa == 0:
        result = math.sqrt(2*math.pi) * sigma
    elif kappa < 0:
        result = math.sqrt(math.pi) * sigma * math.gamma((-1+kappa) / (2*kappa)) / float(math.sqrt(-1*kappa) * math.gamma(1 - (1 / (2*kappa))))
    else:
        result = math.sqrt(math.pi) * sigma * math.gamma(1 / (2*kappa)) / float(math.sqrt(kappa) * math.gamma((1+kappa)/(2*kappa)))
  
    return result


def norm_multi_coupled(std: [float, Any],  kappa: float = 0.0, alpha: int = 2
                       ) -> [float, Any]:

    assert alpha == 1 or alpha == 2, "alpha must be an int and equal to either" + \
                                     " 1 (Pareto) or 2 (Gaussian)."
    
    dim = 1 if isinstance(std, float) else len(std[0])
    if alpha == 1:
        input = (std**0.5)/(1 + (-1 + dim)*kappa)
    else:  # alpha == 2
        gamma_num = math.gamma((1 + (-1 + dim)*kappa)/(2*kappa))
        gamma_dem = math.gamma((1 + dim*kappa)/(2*kappa))
        input = (((math.sqrt(math.pi)*std**0.5)*gamma_num) / (math.sqrt(kappa)*gamma_dem)) 
    return input  # currently wrong ...


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


def coupled_product(x):
    pass


def coupled_sum(x):
    pass


def coupled_sine(x):
    pass


def coupled_cosine(x):
    pass


def weighted_generalized_mean(x):
    pass


def coupled_box_muller(x):
    pass
