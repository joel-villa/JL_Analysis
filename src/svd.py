"""
Has some runnable methods, which plot the convergence of a JL-enhanced power 
iteration (SVD?) method
"""
from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.Plotter import Plotter

import numpy as np
from .tests.svd_tests import *
from .util.eig_functs import *

def test(funct, plotter, mat_name, seed, num_avg, num_iter, args={}):
    """
    Test the given function from tests.py 
    
    funct   - a fucntion which takes a matrix, some x vals, and a seed, and 
              returns xs and ys to be plotted
    plotter - a Plotter() object
    mat    - a matrix name
    seed    - for randomized reproducability
    num_avg - average of how many tests?
    input   - input for funct
    """
    ss_getter = SSGetter(in_csr=False)
    A = ss_getter.get(mat_name)

    print(mat_name)
    ys = np.zeros(num_iter)

    u_star =  top_left(A)
    
    for i in range(num_avg):
        seed_i = seed + i 

        xs, ys_i, label = funct(A, u_star, num_iter, seed=seed_i, **args)
        ys += ys_i
    
    ys = ys / num_avg

    plotter.add_to_plot(xs, ys, label=f"{label} ({mat_name})")
    print("Finished test")

if __name__ == "__main__":
    #TODO: AHHHHH, please work gosh dang it man (debug this, and make cleaner)
    plotter = Plotter(save_fig=True, show_fig=True)
    # mats    = ["494_bus"]
    seed    = 10
    num_avg = 1
    num_iter = 32

    plotter.init_plot("SVD Convergence", "number of iterations", "residual", "svd_convergence") 
    
    test(baseline_svd_convergence, plotter, "494_bus", seed, num_avg, num_iter)
    # test(baseline_svd_convergence, plotter, "1138_bus", seed, num_avg, num_iter)
    # test(baseline_svd_convergence, plotter, "662_bus", seed, num_avg, num_iter)
    test(jl_reduced_svd_convergence, plotter, "494_bus", seed, num_avg, num_iter, {"d" : 400})
    test(jl_reduced_svd_convergence, plotter, "494_bus", seed, num_avg, num_iter, {"d" : 300})
    test(jl_reduced_svd_convergence, plotter, "494_bus", seed, num_avg, num_iter, {"d" : 200})
    test(jl_reduced_svd_convergence, plotter, "494_bus", seed, num_avg, num_iter, {"d" : 100})

    plotter.finish()