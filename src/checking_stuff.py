checking_stuff.py
from numpy import eigs

def eigenval_difference(A,B):
        '''
        Return the 2 norm of the difference of the top eigenvectors of the 
        matrix A and it's sparsified counterpart 

        A        - original matrix
        B        - lower dim matrix

        Return: None if eigs does not converge, 2 norm of difference of top
                eigenvectors otw
        '''

        # Random number generator:
        r_gen = np.random.default_rng()

        n, _ = A.shape

        # Choose the initial vector x, 1 by n
        # Initial guess is close to zero 
        d = B.shape[0]
        v0 = r_gen.normal(loc=0.0, scale=0.01, size=n) 
        v1 = r_gen.normal(loc=0.0,scale=0.01, size=d)

        A_val, _ = eigs(A, k=1, v0=v0) #k = 1 -> only get top eigenvector
        B_val, _ = eigs(B, k=1, v0=v1)
            
        return abs(B_val-A_val)
        """try:
            A_val, _ = eigs(A, k=1, v0=v0) #k = 1 -> only get top eigenvector
            B_val, _ = eigs(B, k=1, v0=v0)
            
            return abs(B_val-A_val)

        except Exception as e:
            # Print exception
            print(f"Error getting top eigenvector of matrix with dimension {B.shape}")
            return None"""
