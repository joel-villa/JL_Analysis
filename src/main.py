from .JohnsonLindenstrauss import JohnsonLindenstrauss
import numpy as np
# from .checking_stuff import *
"""
TODO: 
(1) Code up some Johnson Lindenstrauss algorithm(s)?
(2) Is the top left eigenvector of Johnson Lindenstrauss similar to original? 
"""

from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.MatrixChecker import MatrixChecker

def test_one():
    # MATS = ["494_bus", "662_bus", "685_bus", "1138_bus", "bcsstk21", "bcsstm25", "bcsstm39", "finan512", "jnlbrng1", "m3plates"]
    MATS = ["662_bus", "685_bus", "1138_bus", "bcsstk21", "bcsstm25", "bcsstm39", "finan512", "jnlbrng1", "m3plates"]
    # MATS = ["m3plates"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)
    eps = 0.5
    d = 500
    jonny = JohnsonLindenstrauss()
    
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            approx,orig = jonny.reduce_dimension(A,x,eps,d)
            approx_norm = np.linalg.norm(approx, ord=2)
            orig_norm = np.linalg.norm(orig,ord=2)
            print(abs(approx_norm - orig_norm)/orig_norm)
    
        
        

if __name__ == "__main__":
    test_one()