""" For generating plots """
import matplotlib.pyplot as plt

def plot1(euc_dict, ds, epsilon, num_iter, dims, nnzs, save_fig=False, show_fig=True):
    """
    For plotting compare_eigenvectors() in Tester
    Plot the data from the given nested dictionary

    res_dict - (key, value) = (name, diffs)
    """
    i = 0
    for name, euc_diffs in euc_dict.items():
        plt.plot(ds, euc_diffs, label=f"{name} ({dims[i][0]}x{dims[i][1]}) nnz = {nnzs[i]}")
        
        i += 1

    title = f"JL Eigenvector Preservation (epsilon = {epsilon}, num_avg={num_iter})"
    
    x_label = "Reduced Dimension (d)"
    y_label = "Difference in Top Eigenvectors"
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.suptitle(title)
    
    plt.tight_layout()

    if save_fig:
        plt.savefig("plots/plot1.jpg")
    if show_fig:
        plt.show()
            
