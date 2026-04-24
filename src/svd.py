"""
Has some runnable methods, which plot the convergence of a JL-enhanced power 
iteration (SVD?) method
"""
from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.Plotter import Plotter

import numpy as np
from .tests.svd_tests import *
from .util.eig_functs import *
from .tests.svd_sparse import sparse_svd
from .tests.subset_svd import percent_subset_svd

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
    ys_i = np.zeros(num_iter)

    u_star =  top_left(A)
    
    for i in range(num_avg):
        seed_i = seed + i 

        rng = np.random.default_rng(seed=seed_i)
        u0 = rng.normal(0,1,np.shape(A)[0])
        u0 = u0 / norm(u0)

        xs, ys_i, label = funct(A, u0, u_star, num_iter, seed=seed_i, **args)
        ys += ys_i
    
    ys = ys / num_avg

    plotter.add_to_plot(xs, ys, label=label)
    print("Finished test")

if __name__ == "__main__":
    plotter = Plotter(save_fig=False, show_fig=True)
    # mats    = ["494_bus"]
    seed    = 10
    num_avg = 1
    num_iter = 64

    # mats = ["494_bus", "1138_bus", "bibd_13_6", "bcsstk08"]
    # mats = ["bcspwr06"]
    # mats = ["bcsstk07", "bcsstk19"]
    #SOME THAT CONVERGE FAST: ["beause", "bibd_13_6"]
    # mats = ["494_bus", "1138_bus", "bcsstk08", "bcsstk07", "bcsstk19", "bcsstm12", "bcspwr06"]
    # mats = ["bcsstk10"]
    mats = ["gr_30_30"]
    # mats = ["bcsstm12", "beause", "bcspwr06"]


    types = ["jl_gaussian", "jl_sparse"]
    ps = [90, 98]

    for mat in mats:
        plotter.init_plot(f"SVD Convergence of {mat}", "number of iterations", "residual", f"{mat}_sparse_swap",grid_on=True) 

        test(baseline_svd_convergence, plotter, mat, seed, num_avg, num_iter)
        
        for p in ps:
            for type in types:
                args1 = {"p": p, "type" : type}
                args2 = {"p": p, "step_size": 8, "type" : type}
                # test(jl_percent_reduced, plotter, mat, seed, num_avg, num_iter, args1)
                # test(multi_jl_p_reduce, plotter, mat, seed, num_avg, num_iter, args2)
                test(percent_subset_svd, plotter, mat, seed, num_avg, num_iter, {"p": p})

        plotter.finish()

