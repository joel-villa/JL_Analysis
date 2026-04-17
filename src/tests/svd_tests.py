"""
For testing the convegence of a JL enhanced svd iteration method

All methods in this file return xs, ys, and labels (for easy plotting)
"""

from scipy.sparse.linalg import svds
from scipy.linalg import norm # 2-norm by default
import numpy as np

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
                    #    v0=v, #TODO
                       maxiter=1, 
                       random_state=seed) #TODO should this be more random (ie: add i to seed)
        
        euc_dist = norm(v - v_star)
        ys[i] = euc_dist
    
    return xs, ys, f"standard svd"
    
    