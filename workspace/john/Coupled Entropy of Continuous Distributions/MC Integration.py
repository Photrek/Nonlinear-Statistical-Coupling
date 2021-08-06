# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 01:52:26 2021
@author: jkcle
"""
import numpy as np
from scipy.stats import multivariate_normal, norm
from datetime import datetime

# Set up distribution parameters.
dim = 10
mu, sigma = 0., 1.
loc = np.repeat(mu, repeats=dim)
scale = np.repeat(sigma, repeats=dim)

# Initialize a multivariate normal distribution.
mvn = multivariate_normal(mean=loc, cov=scale)
# Initialize a univariate normal distribution.
normal = norm(loc=mu, scale=sigma)

def mc_integrator(distribution, dim, support, size=1000, seed=0):
    """
    
    Parameters
    ----------
    distribution : function
        A probability density function.
    dim : int
        The number of dimensions of the distribution.
    support : list
        List of the low and high values of the hypercube to integrate over.
    size : int, optional
        Number of samples used for estimation. The default is 1000.
    seed : int, optional
        A random seed for reproducibility. The default is 0.
    Returns
    -------
    float
        The estimate of the integral over the hypercube.
    """
    # Set the random seed.
    np.random.seed(seed)
    # Separate the elements of the support.
    a, b = support[0], support[1]
    # Calculate the volume of the hypercube.
    volume = (b-a)**dim
    # Generate random samples of the appropriate shape.
    samples = np.random.uniform(low=a, high=b, size=(size,dim))
    # Return the estimate of the integral.
    return volume*np.mean(distribution(samples))

# Set the number of samples to use for estimation.
size = 1000000
# Set the low and high value over each dimension of the hypercube.
support = [-4, 4]

# Print the estimate of the integral.
start_time = datetime.now()
mc_estimate = mc_integrator(mvn.pdf, dim, support, size=size)
print(datetime.now() - start_time)
print(mc_estimate)

# Print the exact value of the integral.
exact_value = (normal.cdf(support[1]) - normal.cdf(support[0]))**dim
print(exact_value)

def integrator(distribution, dim, support, size=1000):
    a, b = support[0], support[1]
    step = (b-a)/size
    samples = np.mgrid[a:b:step]
    return (b-a)*np.mean(distribution(samples))


#print(integrator(mvn.pdf, dim, support, size=1000))