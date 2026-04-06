from .JohnsonLindenstrauss import JohnsonLindenstrauss
from .Tester import Tester
import numpy as np
from scipy import linalg
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

    return jonny
def eig_top_right(A):
  eigenvalues,a_ev = linalg.eig(A)
  # Get index of largest eigenvalue (by magnitude)
  top_idx = np.argmax(np.abs(eigenvalues))
  # Top right eigenvector
  a_ev_tr = a_ev[:, top_idx]
  return a_ev_tr

def test_two(eps, d):
    MATS = ["662_bus"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)
    jonny = test_one(eps,d)
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            P = jonny.jl_matrix() 
            approx = P @ A @ P.transpose()
            orig = A
            print(f"approx: {approx.shape[0]}x{approx.shape[0]}")
            print(f"orig: {orig.shape[0]}x{orig.shape[0]}")
            approx_eigenvalues = eig_top_right(approx)
            orig_eigenvalues = eig_top_right(orig.todense())

            #project eigenvalues
            lowdimdiff = (P @ orig_eigenvalues) - approx_eigenvalues
            highdimdiff = orig_eigenvalues - (approx_eigenvalues @P)
            lowdimdiffnorm = np.linalg.norm(lowdimdiff, ord=2)
            highdimdiffnorm = np.linalg.norm(highdimdiff, ord=2)
            print(f'lowdim norm of eig error {lowdimdiffnorm}')
            print(f'highdim norm of eig error {highdimdiffnorm}')

    return jonny

def sparsen(A,density):
  import random
  mat = A.copy()
  nz = 0
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      r = random.uniform(0,1)
      if r > density:
        mat[i,j] =0
        nz += 1
  nnz = mat.shape[0]*mat.shape[1] - nz
  return mat,nnz

def test_three(eps, d):
    MATS = ["662_bus"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)
    jonny = test_two(eps,d)
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        print(f" A nnz {A.nnz}, A entries {A.shape[0]*A.shape[1]}")
        density = A.nnz/(A.shape[0] * A.shape[1])

        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            P = jonny.jl_matrix() 
            approx = P @ A @P.transpose()
            sparse,sparsennz = sparsen(approx, density)
            
            print(f" sparsified nnz {sparsennz}, A entries {sparse.shape[0]*sparse.shape[1]}")
            approx_norm = np.linalg.norm(approx-sparse, ord=2)
            print(f"norm of PA - sparsen(PA) {approx_norm}")

            a_ev_tr = eig_top_right(approx)
            
            s_ev_tr = eig_top_right(sparse)
            
            eigendiff = a_ev_tr - s_ev_tr
            eigendiff_norm = np.linalg.norm(eigendiff, ord=2)
            print(f"norm of eig(PAP^T) - eig(sparsen(PAP^T)) {eigendiff_norm}")
            orig = A
            orig_eigenvalues = eig_top_right(orig.todense())
            norm_oe = np.linalg.norm(orig_eigenvalues,ord=2)

            #project eigenvalues
            lowdimdiff = (P @ orig_eigenvalues) - s_ev_tr
            highdimdiff = orig_eigenvalues - (P.transpose()@ s_ev_tr)
            lowdimdiffnorm = np.linalg.norm(lowdimdiff, ord=2)
            highdimdiffnorm = np.linalg.norm(highdimdiff, ord=2)
            print(f'lowdim norm of P*eig(A) - eig(sparse) {lowdimdiffnorm}')
            print(f'highdim norm of eig(A) - P^T*eig(sparse) {highdimdiffnorm}')
            print(f'highdim norm {highdimdiffnorm} / norm(eig(A)) {norm_oe}={highdimdiffnorm/norm_oe}')
    return jonny


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
    jl = JohnsonLindenstrauss()
    epsilon = 1/64
    n = 8
    ds = [2, 4, 8, 16, 32, 64, 128]
    test = Tester(jl, save_fig=False, show_fig=True)
    test.compare_eigenvectors(ds, epsilon, n)
