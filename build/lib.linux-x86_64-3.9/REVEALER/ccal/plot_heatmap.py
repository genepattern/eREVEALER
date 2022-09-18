from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, show, subplot
import matplotlib.pyplot as plt
from numpy import array, unique
from pandas import DataFrame, Series, isnull
from seaborn import heatmap

from REVEALER.ccal.assign_colors import assign_colors
from REVEALER.ccal.decorate import decorate
from REVEALER.ccal.cluster_2d_array_rows import cluster_2d_array_rows
from REVEALER.ccal.normalize_2d_array import normalize_2d_array
from REVEALER.ccal.save_plot import save_plot
from REVEALER.ccal.style import (CMAP_BINARY_WB, CMAP_CATEGORICAL_TAB20,
                    CMAP_CATEGORICAL_TAB20B, CMAP_CONTINUOUS_BWR, FIGURE_SIZE,
                    FONT_SMALLER, FONT_STANDARD)


def plot_heatmap(df,
                 vmin=None,
                 vmax=None,
                 center=None,
                 robust=False,
                 annotate='auto',
                 fmt='.2g',
                 annot_kws=None,
                 linewidths=0,
                 linecolor='white',
                 square=False,
                 mask=None,
                 drop_axis=None,
                 normalization_axis=None,
                 normalization_method=None,
                 max_std=3,
                 data_type='continuous',
                 axis_to_sort=None,
                 cluster=False,
                 row_annotation=(),
                 column_annotation=(),
                 annotation_colors=(),
                 figure_size=FIGURE_SIZE,
                 cmap=None,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 xlabel_kwargs={},
                 ylabel_kwargs={},
                 xticks=None,
                 yticks=None,
                 xticklabels_kwargs={},
                 yticklabels_kwargs={},
                 file_path=None,
                 **kwargs):
    """
    Plot heatmap.
    """

    if normalization_method:
        df = DataFrame(
            normalize_2d_array(
                df.values, normalization_method, axis=normalization_axis),
            index=df.index,
            columns=df.columns)
        if normalization_method == '-0-':
            df = df.clip(-max_std, max_std)

    if not cmap:
        if data_type == 'continuous':
            cmap = CMAP_CONTINUOUS_BWR
        elif data_type == 'categorical':
            cmap = CMAP_CATEGORICAL_TAB20
        elif data_type == 'binary':
            cmap = CMAP_BINARY_WB
        else:
            raise ValueError('Unknown data_type: {}.'.format(data_type))

    if annotate == 'auto':
        if all([n < 30 for n in df.shape]):
            annotate = True
        else:
            annotate = False

    # Use row and/or column annotation
    if len(row_annotation) or len(column_annotation):
        # Use row annotation
        if len(row_annotation):

            if isinstance(row_annotation, Series):
                row_annotation = row_annotation.copy()
                if not len(row_annotation.index & df.index):
                    # Is Series, but without proper index
                    row_annotation.index = df.index
            else:
                row_annotation = Series(row_annotation, index=df.index)

            row_annotation.sort_values(inplace=True)
            df = df.loc[row_annotation.index]

            # Map non-numerical objects to numbers
            row_o_to_int = {}
            row_int_to_o = {}
            for i, o in enumerate(row_annotation.unique()):
                row_o_to_int[o] = i
                row_int_to_o[i] = o
            row_annotation = row_annotation.map(row_o_to_int)

        # Use column annotation
        if len(column_annotation):

            if isinstance(column_annotation, Series):
                column_annotation = column_annotation.copy()
                if not len(column_annotation.index & df.columns):
                    # Is Series, but without proper index
                    column_annotation.index = df.columns
            else:
                column_annotation = Series(column_annotation, index=df.columns)

            column_annotation.sort_values(inplace=True)
            df = df[column_annotation.index]

            # Map non-numerical objects to numbers
            column_o_to_int = {}
            column_int_to_o = {}
            for i, o in enumerate(column_annotation.unique()):
                column_o_to_int[o] = i
                column_int_to_o[i] = o
            column_annotation = column_annotation.map(column_o_to_int)

    elif axis_to_sort in (0, 1):
        # Copy
        a = array(df)
        a.sort(axis=axis_to_sort)
        df = DataFrame(a, index=df.index)

    elif cluster:
        row_indices = cluster_2d_array_rows(df.values)
        column_indices = cluster_2d_array_rows(df.values.T)
        df = df.iloc[row_indices, column_indices]

        if len(row_annotation):
            row_annotation = row_annotation[row_indices]
        if len(column_annotation):
            column_annotation = column_annotation[column_indices]

    # Set figure
    figure(figsize=figure_size)

    gridspec = GridSpec(10, 10)

    ax_top = subplot(gridspec[0:1, 2:-2])
    ax_center = subplot(gridspec[1:8, 2:-2])
    ax_bottom = subplot(gridspec[8:10, 2:-2])
    ax_left = subplot(gridspec[1:8, 1:2])
    ax_right = subplot(gridspec[1:8, 8:9])

    ax_left.axis('off')
    ax_right.axis('off')
    ax_top.axis('off')
    ax_bottom.axis('off')

    # Get not-nan values for computing min, mean, and max
    values = unique(df.values)
    values = values[~isnull(values)]

    if vmin is None:
        vmin = values.min()
    vmean = values.mean()
    if vmax is None:
        vmax = values.max()

    df = df.astype(float)
    
    heatmap(
        df,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        annot=annotate,
        fmt=fmt,
        annot_kws=annot_kws,
        linewidths=linewidths,
        linecolor=linecolor,
        cbar=False,
        square=square,
        ax=ax_center,
        mask=mask,
        **kwargs)

    # Make legends
    if data_type == 'continuous':
        # Plot colorbar

        # Set colorbat min, mean, and max
        cax, kw = make_axes(
            ax_bottom,
            location='bottom',
            fraction=0.088,
            cmap=cmap,
            norm=Normalize(vmin, vmax),
            ticks=[vmin, vmean, vmax])
        ColorbarBase(cax, **kw)
        decorate(ax=cax)

    elif data_type in ('categorical', 'binary'):
        # Plot category legends

        ax_bottom.axis([0, 1, 0, 1])

        if len(values) <= 20:

            colors = assign_colors(values, cmap=cmap)

            for i, v in enumerate(values):

                x = (i + 1) / (len(values) + 1)

                c = colors[v]

                ax_bottom.plot(
                    x,
                    0.5,
                    'o',
                    color=c,
                    markersize=16,
                    aa=True,
                    clip_on=False)

                ax_bottom.text(
                    x,
                    0.18,
                    v,
                    rotation=90,
                    horizontalalignment='center',
                    verticalalignment='center',
                    clip_on=False,
                    **FONT_STANDARD)

    decorate(
        ax=ax_center,
        despine_kwargs=dict(left=True, bottom=True),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlabel_kwargs=xlabel_kwargs,
        ylabel_kwargs=ylabel_kwargs,
        xticks=xticks,
        yticks=yticks,
        xticklabels_kwargs=xticklabels_kwargs,
        yticklabels_kwargs=yticklabels_kwargs)

    if len(row_annotation):

        if len(annotation_colors):
            cmap = ListedColormap(annotation_colors)
        else:
            cmap = CMAP_CATEGORICAL_TAB20B
        heatmap(
            DataFrame(row_annotation).values,
            ax=ax_right,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            cmap=cmap)

        # Add text
        prev_int = row_annotation.iloc[0]
        prev_y = 0
        for y, int_ in enumerate(row_annotation):
            if prev_int != int_:
                ax_right.text(
                    1.18,
                    prev_y + (y - prev_y) / 2,
                    row_int_to_o[prev_int],
                    horizontalalignment='left',
                    verticalalignment='center',
                    clip_on=False,
                    **FONT_SMALLER)

                prev_int = int_
                prev_y = y

        ax_right.text(
            1.18,
            prev_y + (y + 1 - prev_y) / 2,
            row_int_to_o[prev_int],
            horizontalalignment='left',
            verticalalignment='center',
            clip_on=False,
            **FONT_SMALLER)

    if len(column_annotation):

        if len(annotation_colors):
            cmap = ListedColormap(annotation_colors)
        else:
            cmap = CMAP_CATEGORICAL_TAB20B
        heatmap(
            DataFrame(column_annotation).T,
            ax=ax_top,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            cmap=cmap)

        # Add text
        prev_int = column_annotation.iloc[0]
        prev_x = 0
        for x, int_ in enumerate(column_annotation):
            if prev_int != int_:
                ax_top.text(
                    prev_x + (x - prev_x) / 2,
                    -0.18,
                    column_int_to_o[prev_int],
                    rotation=90,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    clip_on=False,
                    **FONT_SMALLER)

                prev_int = int_
                prev_x = x

        ax_top.text(
            prev_x + (x + 1 - prev_x) / 2,
            -0.18,
            column_int_to_o[prev_int],
            rotation=90,
            horizontalalignment='center',
            verticalalignment='bottom',
            clip_on=False,
            **FONT_SMALLER)

    if file_path:
        save_plot(file_path)

    plt.close()
