# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable
from scipy.integrate import nquad
from .entropy import coupled_entropy


def shannon_entropy(density_func: Callable[..., np.ndarray],
                    dim: int = 1,
                    support: list = [[-np.inf, np.inf]],
                    root = False
                    ) -> [float, np.ndarray]:
    """
    Add description here.

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

    return coupled_entropy(density_func,
                           kappa=0.0, 
                           alpha=alpha, 
                           dim=dim, 
                           support=support,
                           root=root
                           )


def tsallis_entropy(density_func: Callable[..., np.ndarray],
                    kappa: float = 0.0,
                    alpha: int = 1,
                    dim: int = 1,
                    support: list = [(-np.inf, np.inf)],
                    normalize = False,
                    root = False
                    ) -> [float, np.ndarray]:
    """
    Add description here.

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
        entropy = (1+kappa)**(1/alpha) * coupled_entropy(density_func,
                                                         kappa=kappa,
                                                         alpha=alpha,
                                                         dim=dim,
                                                         support=support,
                                                         root=root
                                                         )
    else:
        def un_normalized_density_func(*args):
            if dim == 1:
                x = np.array(args)
            else:
                x = np.array([args]).reshape(1, dim)
            return density_func(x)**(1+(alpha*kappa/(1+kappa)))

        entropy = (nquad(un_normalized_density_func, support)[0]
                       * (1+kappa)**(1/alpha)
                       * coupled_entropy(density_func,
                                         kappa=kappa,
                                         alpha=alpha,
                                         dim=dim,
                                         support=support,
                                         root=root
                                         )
                       )

    return entropy