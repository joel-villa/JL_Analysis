"""
A dimensionality reduction matrix
"""
from numpy import log 
import math

class JohnsonLindenstrauss:
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

        numerator = 8 * log(n)
        denominator = epsilon * epsilon
        return math.ceil(numerator / denominator)
    
    def jl_matrix(self, n, d):
        """
        Randomly generate a matrix of dimensions (dxn) in the way outlined in
        section 4.4 of the following lecture notes: 
        https://www.cs.unm.edu/~saia/classes/506-s26/lec/HighDim+JLProjection.pdf
        """

        #TODO 

    def reduce_dimension(self, A, epsilon, d=None):
        """
        Reduce the dimension of A as much as possible given some epsilon,
        if the new dimension is not given, assume reducing as extremely as 
        possible. If new dimension provided, check it is valid, if not print a 
        warning

        A       - original matrix (nxn)
        epsilon - some factor of allowed error
        d       - the new dimension of matrix

        RETURN: a new matrix of dimension dxn
        """

        if not self.valid_epsilon(epsilon=epsilon):
            print(f"WARNING: invalid epsilon {epsilon} not in range (0, 1)")

        d_lower_bound = self.dimension_lower_bound(A.shape[0], epsilon=epsilon)

        if d is None:
            # No d provided, autogenerate based on A's dimensions and epsilon
            d = d_lower_bound
        elif d_lower_bound > d:
            print(f"WARNING: d_lower_bound = {d_lower_bound} > {d} = d")
        
        return A @ self.jl_matrix(A.shape[0], d)