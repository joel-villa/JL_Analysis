"""
Joel's main
"""

from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.Plotter import Plotter
from .tests import *

import matplotlib.pyplot as plt

import numpy as np

def test_one():
    """
    For testing preservation of top eigenvector w/ tests from tests.py
    """
    ss_getter = SSGetter(in_csr=False)
    mats = ss_getter.get_by_name(names=["1138_bus"])

    seed = 10
    num_avg = 1
    xs = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


    plotter = Plotter(save_fig=False, show_fig=True)
    plotter.init_plot(title="top eigenvector preservation of normalized JL",
                      x_label="epsilon",
                      y_label="norm of difference in top eigenvectors",
                      save_name="normalized_JL")
    
    # Gaussian jl test
    print("Starting gaussian test")
    for name, A in mats.items():
        ys = np.zeros(np.shape(xs))

        for i in range(num_avg):
            seed_i = seed + i 
            xs, ys_i = test_jl_top_eig_pres(A, xs, seed=seed_i)
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

if __name__ == '__main__':
    test_two()