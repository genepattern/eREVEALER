from os.path import isfile

from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt

from REVEALER.ccal.path import establish_path


def save_plot(file_path, overwrite=True, dpi=100):
    """
    Establish file path and save plot.
    Arguments:
        file_path (str):
        overwrite (bool):
        dpi (int):
    Returns:
        None
    """

    # If the figure doesn't exist or overwriting
    if not isfile(file_path) or overwrite:

        establish_path(file_path)

        savefig(file_path, dpi=dpi, bbox_inches='tight')

        plt.close()
