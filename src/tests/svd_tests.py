"""
For testing the convegence of a JL enhanced svd iteration method

All methods in this file return xs, ys, and labels (for easy plotting)
"""

from scipy.sparse.linalg import svds
from scipy.linalg import norm # 2-norm by default

import numpy as np

from ..util.eig_functs import euclidean_dist
from ..util.scikit_jl import jl_gaussian
from ..util.svd_util import topsing
from ..util.svd_util import v_from_u

def baseline_svd_convergence(A, u_0, u_star, num_iter, seed):
    """
    The baseline SVD convergence
    """

    v =  v_from_u(A, u_0)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = euclidean_dist(u_0, u_star)
    xs[0] = 0

    print(f"ys[0] = {ys[0]}")


    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, _, v = topsing(v0=v,
                          A=A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    # print(f"xs: {xs}")
    # print(f"ys: {ys}")

    print(f"ys[0] = {ys[0]}")
    
    return xs, ys, f"standard svd {A.shape}"

def jl_reduced_svd_convergence(A, u_0, u_star, num_iter, seed, d):
    """
    Convergence of SVD on a JL-dimensionally reduced version of A
    """

    

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    reduced_A = jl_gaussian(A, d=d, seed=seed, eps=0.99)

    v =  v_from_u(reduced_A, u_0)

    ys[0] = euclidean_dist(u_0, u_star)
    print(f"ys[0] = {ys[0]}")
    xs[0] = 0

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = topsing(v0=v,
                          A=reduced_A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    # print(f"xs: {xs}")
    # print(f"ys: {ys}")

    print(f"ys[0] = {ys[0]}")
    
    return xs, ys, f"jl-reduced svd {reduced_A.shape}"

# def svds_convergence(A, v0, v_star, num_iter, seed):
#     """
#     The baseline SVD convergence
#     NOTE: svds() from scipy.sparse.linalg is not an itterative method
#     """
#     v = v0.copy()

#     xs = np.zeros(num_iter)
#     ys = np.zeros(num_iter)

#     for i in range(num_iter):
#         # NOTE: using scikit-learn -> top left eig is of significance
#         v, _, _ = svds(A, 
#                        k=1,        # top eigenvector
#                        which='LM', # top eigenvector
#                        v0=v, #TODO
#                        maxiter=1, 
#                        random_state=seed) #TODO should this be more random (ie: add i to seed)
        
#         v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

#         euc_dist = euclidean_dist(v, v_star)
#         ys[i] = euc_dist
#         xs[i] = i

#     # print(f"xs: {xs}")
#     # print(f"ys: {ys}")
    
#     return xs, ys, f"standard svd"
    
    