"""
For SVD functions
"""

import numpy as np
from scipy.linalg import norm # 2-norm by default


# def iterative_top_svd(A, v0, max_iter=100):
    
#     # Initialize a random vector
#     v = v0.copy()
#     v /= np.linalg.norm(v)
    
#     for _ in range(max_iter):
#         # Power step: v = (A^T * A) * v
#         v_new = A.T @ (A @ v)
#         v_new_norm = np.linalg.norm(v_new)
#         v = v_new / v_new_norm
        
#     # Compute singular value and left vector
#     sigma = np.linalg.norm(A @ v)
#     u = (A @ v) / sigma
#     return u, sigma, v

def topsing(x0, A, maxiter=10):
    """
    x0      - an initial guess for a vector w/ the same number of columns as A
    A       - A matrix
    maxiter - how many iterations of SVD? 
    Adapted from section "4.4.2. Computing the top singular vector", found here:
    https://mmids-textbook.github.io/chap04_svd/04_power/roch-mmids-svd-power.html
    """
    x = x0.copy()
    B = A.T @ A 
    for _ in range(maxiter):
        x = B @ x
    v = x / norm(x)
    s = norm(A @ v)
    u = A @ v / s
    return u, s, v