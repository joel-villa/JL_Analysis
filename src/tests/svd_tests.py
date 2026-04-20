"""
For testing the convegence of a JL enhanced svd iteration method

All methods in this file return xs, ys, and labels (for easy plotting)
"""

from scipy.sparse.linalg import svds
from scipy.linalg import norm # 2-norm by default

import numpy as np

from ..util.eig_functs import euclidean_dist
from ..util.svd_util import topsing

def baseline_svd_convergence(A, v0, v_star, num_iter, seed):
    """
    The baseline SVD convergence
    """
    v = v0.copy()

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    for i in range(num_iter):
        # NOTE: using scikit-learn -> top left eig is of significance
        v, _, _ = svds(A, 
                       k=1,        # top eigenvector
                       which='LM', # top eigenvector
                       v0=v, #TODO
                       maxiter=1, 
                       random_state=seed) #TODO should this be more random (ie: add i to seed)
        
        v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = euclidean_dist(v, v_star)
        ys[i] = euc_dist
        xs[i] = i

    # print(f"xs: {xs}")
    # print(f"ys: {ys}")
    
    return xs, ys, f"standard svd"
    
    