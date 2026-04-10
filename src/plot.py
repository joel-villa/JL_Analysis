""" For generating plots """
import matplotlib.pyplot as plt

def plot1(euc_dict, cos_dict, ds, epsilon, num_iter, save_fig=False, show_fig=True):
    """
    For plotting compare_eigenvectors() in Tester
    Plot the data from the given nested dictionary

    res_dict - (key, value) = (name, diffs)
    """

    i = 0
    for name, euc_diffs in euc_dict.items():
        # One row, two columns of subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))

        cos_diffs = cos_dict[name]

        title = f"JL of {name} (epsilon = {epsilon}, num_avg={num_iter})"
        
        x_label = "Reduced Dimension (d)"
        y_label = "Difference in Top Eigenvectors"

        # Euclidean Plot
        ax1.plot(ds, euc_diffs)
        ax1.set_title("Euclidean Distance")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
            
        # Cosine Plot
        ax2.plot(ds, cos_diffs)
        ax2.set_title("Cosine Distance")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)

        fig.suptitle(title)

        plt.tight_layout()

        if save_fig:
            plt.savefig("plots/" + name  + ".jpg")
        if show_fig:
            plt.show()
            
        i += 1