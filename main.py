import JohnsonLindenstrauss as JL
import numpy as np
"""
TODO: 
(1) Code up some Johnson Lindenstrauss algorithm(s)?
(2) Is the top left eigenvector of Johnson Lindenstrauss similar to original? 
"""

from Sparsification_Research.SSGetter import SSGetter
import Sparsification_Research.MatrixChecker as MC

def test_one():
    MATS = ["494_bus", "662_bus", "685_bus", "1138_bus", "bcsstk21", "bcsstm25", "bcsstm39", "finan512", "jnlbrng1", "m3plates"]
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)
    eps = 0.5
    d = 500
    jonny = JL.JohnsonLindenstrauss()
    manny = MC.MatrixChecker()
    for name, A in mats.items():
        print(f"{name}: {A.shape[0]}x{A.shape[0]}")
        B = jonny.reduce_dimension(A,eps,d)
        B = np.transpose(B)
        B = jonny.reduce_dimension(B,eps,d)
        print(manny.eigenval_difference(A,B))
    
        
        

if __name__ == "__main__":
    test_one()