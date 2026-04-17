"""
For testing the convegence of a JL enhanced svd iteration method

All methods in this file return xs and ys (for easy plotting)
"""

from scipy.sparse.linalg import svds

def baseline_svd_convergence():
    """
    The baseline SVD convergence
    """