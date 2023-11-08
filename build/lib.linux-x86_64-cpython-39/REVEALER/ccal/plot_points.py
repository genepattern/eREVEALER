from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt

from REVEALER.ccal.decorate import decorate
from REVEALER.ccal.save_plot import save_plot
from REVEALER.ccal.style import FIGURE_SIZE


def plot_points(*args,
                ax=None,
                figure_size=FIGURE_SIZE,
                title='',
                xlabel='',
                ylabel='',
                file_path=None,
                **kwargs):
    """
    Plot points.
    """

    if ax:
        save_and_show = False
    else:
        ax = figure(figsize=figure_size).gca()
        save_and_show = True

    for k, v in [
        ('linestyle', ''),
        ('marker', '.'),
        ('color', '#20D9BA'),
    ]:
        if k not in kwargs:
            kwargs[k] = v

    ax.plot(*args, **kwargs)

    decorate(title=title, xlabel=xlabel, ylabel=ylabel)

    if save_and_show:

        if file_path:
            save_plot(file_path)

        plt.close()
