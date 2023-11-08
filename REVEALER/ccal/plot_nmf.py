from matplotlib.pyplot import figure

from REVEALER.ccal.plot_heatmap import plot_heatmap
from REVEALER.ccal.style import FIGURE_SIZE
from REVEALER.ccal.path import establish_path


def plot_nmf(w_matrix,
             h_matrix,
             max_std=3,
             figure_size=FIGURE_SIZE,
             file_path_prefix=None):
    """
    Plot NMF.
    """

    if file_path_prefix:
        establish_path(file_path_prefix)
        w_file_path = '{}_w.png'.format(file_path_prefix)
        h_file_path = '{}_h.png'.format(file_path_prefix)
    else:
        w_file_path = h_file_path = None

    figure(figsize=figure_size)

    plot_heatmap(
        w_matrix,
        normalization_method='-0-',
        normalization_axis=0,
        max_std=max_std,
        cluster=True,
        figure_size=(max(figure_size), max(figure_size) * 0.8),
        title='NMF W Matrix for k={}'.format(w_matrix.shape[1]),
        xlabel='Component',
        ylabel='Feature',
        file_path=w_file_path)

    plot_heatmap(
        h_matrix,
        normalization_method='-0-',
        normalization_axis=1,
        max_std=max_std,
        cluster=True,
        figure_size=(max(figure_size), max(figure_size) * 0.8),
        title='NMF H Matrix for k={}'.format(h_matrix.shape[0]),
        xlabel='Sample',
        ylabel='Component',
        file_path=h_file_path)
