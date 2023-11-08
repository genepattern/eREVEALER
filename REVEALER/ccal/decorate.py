from matplotlib.pyplot import gca, sca, suptitle
from seaborn import despine, set_style

from REVEALER.ccal.style import FONT_LARGER, FONT_LARGEST, FONT_SMALLER, FONT_STANDARD
from REVEALER.ccal.dict_ import merge_dicts_with_function


def decorate(ax=None,
             style='ticks',
             despine_kwargs={},
             title=None,
             title_kwargs={},
             yaxis_position='left',
             xlabel=None,
             ylabel=None,
             xlabel_kwargs={},
             ylabel_kwargs={},
             xticks=None,
             yticks=None,
             max_n_xticks=None,
             max_n_yticks=None,
             max_xtick_size=None,
             max_ytick_size=None,
             xticklabels_kwargs={},
             yticklabels_kwargs={},
             legend_loc='best'):
    """
    Decorate an ax.
    """

    if ax:
        sca(ax)
    else:
        ax = gca()

    if legend_loc:
        ax.legend(
            loc=legend_loc,
            prop={
                'size': FONT_STANDARD['fontsize'],
                'weight': FONT_STANDARD['weight'],
            },
            labels=['values'])

    # Set plot style
    set_style(style)
    despine(**despine_kwargs)

    # Title
    if title:
        title_kwargs = merge_dicts_with_function(FONT_LARGEST, title_kwargs,
                                                 lambda a, b: b)
        suptitle(title, **title_kwargs)

    # Set y axis position
    if yaxis_position == 'right':
        ax.yaxis.tick_right()

    # Style x label
    if xlabel is None:
        xlabel = ax.get_xlabel()
    xlabel_kwargs = merge_dicts_with_function(FONT_LARGER, xlabel_kwargs,
                                              lambda a, b: b)
    ax.set_xlabel(xlabel, **xlabel_kwargs)

    # Style y label
    if ylabel is None:
        ylabel = ax.get_ylabel()
    ylabel_kwargs = merge_dicts_with_function(FONT_LARGER, ylabel_kwargs,
                                              lambda a, b: b)
    ax.set_ylabel(ylabel, **ylabel_kwargs)

    # Style x ticks
    if xticks is not None:
        ax.set_xticks(xticks)

    if max_n_xticks and max_n_xticks < len(ax.get_xticks()):
        ax.set_xticks([])

    # Style x tick labels
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    if len(xticklabels):

        if xticklabels[0]:
            # Limit tick label size
            if max_xtick_size:
                xticklabels = [t[:max_xtick_size] for t in xticklabels]
        else:
            xticklabels = ax.get_xticks()

        # Set tick label rotation
        if 'rotation' not in xticklabels_kwargs:
            xticklabels_kwargs['rotation'] = 90

        xticklabels_kwargs = merge_dicts_with_function(
            FONT_SMALLER, xticklabels_kwargs, lambda a, b: b)

        ax.set_xticklabels(xticklabels, **xticklabels_kwargs)

    # Style y ticks
    if yticks is not None:
        ax.set_yticks(yticks)

    if max_n_yticks and max_n_yticks < len(ax.get_yticks()):
        ax.set_yticks([])

    # Style y tick labels
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    if len(yticklabels):

        if yticklabels[0]:
            # Limit tick label size
            if max_ytick_size:
                yticklabels = [t[:max_ytick_size] for t in yticklabels]
        else:
            yticklabels = ax.get_yticks()

        # Set tick label rotation
        if 'rotation' not in yticklabels_kwargs:
            yticklabels_kwargs['rotation'] = 0

        yticklabels_kwargs = merge_dicts_with_function(
            FONT_SMALLER, yticklabels_kwargs, lambda a, b: b)

        ax.set_yticklabels(yticklabels, **yticklabels_kwargs)
