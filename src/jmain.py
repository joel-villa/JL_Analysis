"""
Joel's main
"""

from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.Plotter import Plotter
from .tests import *

import matplotlib.pyplot as plt

import numpy as np

def test(funct, plotter, mats, seed, num_avg, input):
    """
    Test the given function from tests.py 
    
    funct   - a fucntion which takes a matrix, some x vals, and a seed, and 
              returns xs and ys to be plotted
    plotter - a Plotter() object
    mats    - a list of matrix names
    seed    - for randomized reproducability
    num_avg - average of how many tests?
    input   - input for funct
    """
    ss_getter = SSGetter(in_csr=False)
    mats = ss_getter.get_by_name(names=mats)
    
    print("Starting test")
    for name, A in mats.items():
        print(name)
        ys = np.zeros(np.shape(input))

        for i in range(num_avg):
            seed_i = seed + i 
            xs, ys_i = funct(A, input, seed=seed_i)
            ys += ys_i

        ys = ys / num_avg

        plotter.add_to_plot(xs, ys, label=f"{name} (nnz = {A.nnz})")
    print("Finished test")
    plotter.finish()


def test_one():
    """
    For testing preservation of top eigenvector w/ tests from tests.py
    """
    ss_getter = SSGetter(in_csr=False)
    mats = ss_getter.get_by_name(names=["1138_bus"])

    seed = 10
    num_avg = 1
    eps = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


    plotter = Plotter(save_fig=False, show_fig=True)
    plotter.init_plot(title="top eigenvector preservation of normalized JL",
                      x_label="epsilon",
                      y_label="norm of difference in top eigenvectors",
                      save_name="normalized_JL")
    
    # Gaussian jl test
    print("Starting gaussian test")
    for name, A in mats.items():
        ys = np.zeros(np.shape(eps))

        for i in range(num_avg):
            seed_i = seed + i 
            xs, ys_i = scikit_jl_top_eig_pres(A, eps, seed=seed_i)
            ys += ys_i

        print(ys)
        ys = ys / num_avg
        print(ys)

        plotter.add_to_plot(xs, ys, label=f"{name} (nnz = {A.nnz})")
        # plt.plot(xs, ys, label=f"{name} (nnz = {A.nnz})")
    print("Finished gaussian test")
    plotter.finish()

def test_two():
    """
    For testing preservation of top eigenvector w/ tests from tests.py
    """
    ss_getter = SSGetter(in_csr=False)
    mats = ss_getter.get_by_name(names=["494_bus", "1138_bus", "ch7-7-b1", "rel5"])

    seed = 10
    num_avg = 1
    xs = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


    plotter = Plotter(save_fig=True, show_fig=True)
    plotter.init_plot(title="top eigenvector preservation of JL",
                      x_label="percentage reduction",
                      y_label="norm of difference in top eigenvectors",
                      save_name="plot_of_eig_pres")

    # Percentage jl test
    print("Starting J percentage test")
    ps = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    for name, A in mats.items():
        print(name)
        ys = np.zeros(np.shape(ps))

        for i in range(num_avg):
            seed_i = seed + i 
            xs, ys_i = jl_top_eig_pres(A, ps, seed=seed_i)
            ys += ys_i

        ys = ys / num_avg

        print(f"ys = {ys}")
        print(f"ps = {ps}")

        plotter.add_to_plot(ps, ys, label=f"{name} (nnz = {A.nnz})")
        # plt.plot(xs, ys, label=f"{name} (nnz = {A.nnz})")
    print("Finished J percentage test")

    # plt.legend()
    # plt.tight_layout()

    # plt.show()
    plotter.finish()
def run_scikit_eig_percent_reduce(plotter, mats, seed, num_avg):
    """
    Given a plotter, run the associated test
    """
    plotter.init_plot(title="top eigenvector preservation of normalized JL",
                      x_label="percent dimensionality reduction",
                      y_label="norm of difference in top eigenvectors",
                      save_name="scikit_eig_percent_reduce")
    function = scikit_eig_percent_reduce

    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    test(funct=function, 
         plotter=plotter,
         mats=mats,
         seed=seed,
         num_avg=num_avg,
         input=ps)


if __name__ == '__main__':
    plotter = Plotter(save_fig=True, show_fig=True)
    # mats    = ["494_bus"]
    mats    = ["494_bus", "1138_bus", "bibd_11_5", "bibd_13_6", "bcsstk08"]
    seed    = 10
    num_avg = 1

    run_scikit_eig_percent_reduce(plotter, mats, seed, num_avg)