# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 02:15:16 2020

@author: jkcle
"""
# -*- coding: utf-8 -*-
import numpy as np
from typing import Any, List  # for NDArray types
import math

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


def coupled_logarithm(values: [int, float, np.ndarray], kappa: float = 0.0, dim: int = 1) -> [float, np.ndarray]:
    """
    Generalization of the logarithm function, which defines smooth
    transition to power functions.

    Parameters
    ----------
    values : [float, Any]
        Input variable in which the coupled logarithm is applied to.
    kappa : float, optional
        Coupling parameter which modifies the coupled logarithm function. 
        The default is 0.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    [float, Any]
        Returns the coupled logarithm of the values.

    """
    
    # Convert number into np.ndarray to keep consistency.
    values = np.array(values) if isinstance(values, (int, float)) else values
    assert isinstance(values, np.ndarray), ("value must be an int, float, or "
                                            "np.ndarray.")
    assert 0. not in values, "value must not be or contain any zero(s)."
    
    # If kappa is 0, return the natural logarithm of the values.
    if kappa == 0:
        coupled_log_values = np.log(values)
    # Otherwise, return (1 / kappa) * (values^(kappa / (1 + dim*kappa)) - 1).
    else:
        coupled_log_values = (1 / kappa) * (values**(kappa / (1 + dim*kappa)) - 1)
        
    return coupled_log_values


def coupled_exponential(values: [float, Any], kappa: float = 0.0, dim: int = 1) -> [float, Any]:
    """
    Generalization of the exponential function.

    Parameters
    ----------
    values : [float, Any]
        Input values in which the coupled exponential is applied to.
    kappa : float, optional
        Coupling parameter which modifies the coupled exponential function. 
        The default is 0.0.
    dim : int, optional
        The dimension of x, or rank if x is a tensor. The default is 1.

    Returns
    -------
    float
        The coupled exponential values.

    """
    
    assert (dim >= 0) & (type(dim) == int), "dim must be a postive integer."
    
    # If kappa is 0, raise e to the power of each of the values.
    if kappa == 0:
        coupled_exp_value = np.exp(values)
    # If kappa is not 0, execute the conditionals below.   
    else:
        # If kappa is positive, raise 1 + kappa*values to the power of
        # 1 / (kappa / (1 + dim*alpha))
        if kappa > 0:
        	coupled_exp_value = ((1 + kappa*values)**(1/(kappa / 
                                                    (1 + dim*kappa))))
        # If kappa is negative, but 1 + kappa*values is non-negative, raise
        # 1 + kappa*values to the power of 1 / (kappa / (1 + dim*alpha))
        elif (1 + kappa*values) >= 0:
                coupled_exp_value = (1+kappa*values)**(1/(kappa/(1+dim*kappa)))
        # If kappa is negative, 1 + kappa*values is negative, and 
        # (kappa / (1 + dim*alpha)) is positive, return 0.
        elif (kappa / (1 + dim*kappa)) > 0: 
                coupled_exp_value = 0
        # Otherwise, return infinity.
        else:
       		coupled_exp_value = float('inf')

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
        coupled_logarithm_dist_q = (1/-alpha)*coupled_logarithm(values=dist_q**(-alpha), 
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
        coupled_logarithm_dist_q = coupled_logarithm(values=dist_q**(-alpha), 
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
                 validate_args: bool = False,
                 allow_nan_stats: bool = True
                 ):
        loc = np.asarray(loc) if isinstance(loc, List) else loc
        scale = np.asarray(scale) if isinstance(scale, List) else scale
        if validate_args:
            assert isinstance(loc, (int, float, np.ndarray)), "loc must be either an int/float type for scalar, or an list/ndarray type for multidimensional."
            assert isinstance(scale, (int, float, np.ndarray)), "scale must be either an int/float type for scalar, or an list/ndarray type for multidimensional."
            assert type(loc) == type(scale), "loc and scale must be the same type."
            if isinstance(loc, np.ndarray):
                assert loc.shape == scale.shape, "loc and scale must have the same dimensions (check respective .shape())."
                assert np.all((scale >= 0)), "All scale values must be greater or equal to 0."            
            else:
                assert scale >= 0, "scale must be greater or equal to 0."            
            assert isinstance(kappa, (int, float)), "kappa must be an int or float type."
            assert isinstance(alpha, int), "alpha must be an int that equals to either 1 or 2."
            assert alpha in [1, 2], "alpha must be equal to either 1 or 2."
        self.loc = loc
        self.scale = scale
        self.kappa = kappa
        self.alpha = alpha
        #print(f"<nsc.distributions.CoupledNormal batch_shape={self._batch_shape()} event_shape={self._event_shape()}>")

    def n_dim(self):
        return 1 if self._event_shape() == [] else self._event_shape()[0]
    
    def _batch_shape(self) -> List:
        if self._rank(self.loc) == 0:
            # return [] signifying single batch of a single distribution
            return []
        elif self.loc.shape[0] == 1:
            # return [] signifying single batch of a multivariate distribution
            return []
        else:
            # return [batch size]
            return [self.loc.shape[0]]

    def _event_shape(self) -> List:
        if self._rank(self.loc) < 2:
            # return [] signifying single random variable (regardless of batch size)
            return []
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
        
        return p
    
    # Normalization of 1-D Coupled Gaussian (NormCG)
    def _normalized_term(self) -> [int, float, np.ndarray]:
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