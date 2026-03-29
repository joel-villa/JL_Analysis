"""
A dimensionality reduction matrix
"""
import numpy as np
import random
import math

class JohnsonLindenstrauss:
    def __init__(self, seed):
        self.seed

    def jl_matrix_product(self, A, d, eps) :

        B = np.zeros(d,A.shape[0])

        lower = -1 * math.sqrt(1/d)
        upper = math.sqrt(1/d)
        options = [lower,upper]
        
        dict = {}

        # for row, col, i in zip(A.row, A.col, range(A.nnzs)) :
        #     #TODO
        #     random.seed(self.get_row_seed(col))
        #     val = A.data[i] * options[self.get_entry(*,row,d)]

        #     if dict.get(col) is None :
        #         dict[col] = val
        #     else :
        #         dict[col] += val

        # for key,value in 
            
            

    def get_row_seed(self, row):
        random.seed(self.seed)
        for i in range(row) : r = random.randint()
        return r
    
    def get_entry(self, row, col, d) :
        random.seed(self.get_row_seed(row))
        options = [0,1]
        for c in range(col) : result = random.choice(options)
        return result
    

    def valid_epsilon(self, epsilon):
        """
        Check if epsilon is valid for the Johnson Lindenstrauss dimensionality 
        reduction method

        RETURN: 0 < epsilon < 1
        """
        return 0 < epsilon and epsilon < 1

    def dimension_lower_bound(self, n, epsilon):
        """
        Get the lower bound of the dimension for the Johnson Lindenstrauss 
        algorithm s.t. 
        (1 - e)||u - v||^2 <= ||f(u) - f(v)||^2 <= (1 + e)||u - v||^2
        via the Johnson Lindenstrauss Lemma
        
        k > 8(ln n) / ε^s

        n       - the original dimension
        epsilon - the amount of allowed error 

        RETURN - the lower bound of the dimension (inclusive)
        """

        if not self.valid_epsilon(epsilon=epsilon):
            print(f"WARNING: invalid epsilon {epsilon} not in range (0, 1)")

        numerator = 8 * np.log(n)
        denominator = epsilon * epsilon
        return math.ceil(numerator / denominator)
    
    def jl_matrix(self, d, n, seed=random.randint):
        """
        Randomly generate a matrix of dimensions (dxn) in the way outlined in
        section 4.4 of the following lecture notes: 
        https://www.cs.unm.edu/~saia/classes/506-s26/lec/HighDim+JLProjection.pdf
        """
        self.seed = seed
        random.seed(seed)
        lower = -1 * math.sqrt(1/d)
        upper = math.sqrt(1/d)
        options = [lower,upper]
        out = np.zeros((d,n))
    
        for i in range(d):
            for j in range(n):
                out[i][j] = random.choice(options)
        return out


    def reduce_dimension(self, A, x, epsilon, d=None):
        """
        Reduce the dimension of A as much as possible given some epsilon,
        if the new dimension is not given, assume reducing as extremely as 
        possible. If new dimension provided, check it is valid, if not print a 
        warning

        A       - original matrix (nxn)
        x       - vector to compare matrix vector product
        epsilon - some factor of allowed error
        d       - the new dimension of matrix

        RETURN: 
            -The approximation of Ax given by P*(Ax)
            -The original Ax for comparison
        """

        if not self.valid_epsilon(epsilon=epsilon):
            print(f"WARNING: invalid epsilon {epsilon} not in range (0, 1)")

        d_lower_bound = self.dimension_lower_bound(A.shape[0], epsilon=epsilon)

        if d is None:
            # No d provided, autogenerate based on A's dimensions and epsilon
            d = d_lower_bound
        elif d_lower_bound > d:
            print(f"WARNING: d_lower_bound = {d_lower_bound} > {d} = d")
        
        JLM = self.jl_matrix(d, A.shape[0])
        
        print(f"JLM dim: {JLM.shape}, A dim: {A.shape}, x dim: {x.shape}")

        original = A @ x
        approx = JLM @ original
        
        return approx, original