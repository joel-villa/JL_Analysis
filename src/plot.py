""" For generating plots """
import matplotlib.pyplot as plt

def plot1(mat_dict, ds, epsilon, num_iter, save_fig=False, show_fig=True):
    """
    For plotting compare_eigenvectors() in Tester
    Plot the data from the given nested dictionary

    res_dict - (key, value) = (name, diffs)
    """

    i = 0
    for name, diffs in mat_dict.items():
        title = f"JL w/ {name} (epsilon = {epsilon}, num_avg={num_iter}"
        
        x_label = "reduced dimension (d)"
        y_label = "Relative difference in top eigenvectors"
        
        plt.plot(ds, diffs)
            
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()

        if save_fig:
            plt.savefig("plots/" + name  + ".jpg")
        if show_fig:
            plt.show()
            
        i += 1