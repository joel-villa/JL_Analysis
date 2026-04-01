from .JohnsonLindenstrauss import JohnsonLindenstrauss
import numpy as np
"""
TODO: 
(1) Code up some Johnson Lindenstrauss algorithm(s)?
(2) Is the top left eigenvector of Johnson Lindenstrauss similar to original? 
"""

from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.MatrixChecker import MatrixChecker

def test_one(eps, d):
    MATS = ["662_bus"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)

    jonny = JohnsonLindenstrauss()
    
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            approx,orig = jonny.get_b(A,x,eps,d)
            approx_norm = np.linalg.norm(approx, ord=2)
            orig_norm = np.linalg.norm(orig,ord=2)
            print(abs(approx_norm - orig_norm)/orig_norm)
    
def jl_validity(eps, d):
    # MATS = ["494_bus", "662_bus", "685_bus", "1138_bus", "bcsstk21", "bcsstm25", "bcsstm39", "finan512", "jnlbrng1", "m3plates"] #some matrices that converged w/ Jacobi
    MATS = ["662_bus"]
    NUM_XS = 10

    
    ssgetter = SSGetter(in_csr=False, row_bounds=(600, 700))

    # mats = ssgetter.get_next(1)
    mats = ssgetter.get_by_name(MATS)
    jonny = JohnsonLindenstrauss()
    
    for name, A in mats.items():
        print(f"{name}: {A.shape[0]}x{A.shape[0]}")
        xs = np.random.rand(NUM_XS, A.shape[0]) # randomly generated x vectors
        A_reduced = jonny.reduce(A, eps, d)

        print(f"A_reduced.shape = {A_reduced.shape}")

        res = []
        for x in xs:
            b_approx = A_reduced @ x
            b = A @ x

            approx_norm = np.linalg.norm(b_approx, ord=2)
            orig_norm = np.linalg.norm(b,ord=2)
            res.append(abs(approx_norm - orig_norm)/orig_norm)   
        print(f"average distance of {NUM_XS} randomly generated vectors {np.mean(res)}")   

if __name__ == "__main__":
    epsilon = 1/64
    d = 500
    test_one(epsilon, d)
    jl_validity(epsilon, d)