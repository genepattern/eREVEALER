from matplotlib.colors import ListedColormap
from numpy import number

from REVEALER.ccal.style import CMAP_CATEGORICAL_TAB20


def assign_colors(objects, cmap=CMAP_CATEGORICAL_TAB20, colors=()):
    """
    Assign colors to objects.
    Arguments:
        objects (iterable):
        cmap (matplotlib.cm):
        colors (iterable):
    Returns:
        dict: {state: color}
    """

    o_to_color = {}

    for i, o in enumerate(sorted(set(objects))):

        if len(colors):

            # Make colormap from colors
            cmap = ListedColormap(colors, N=len(colors))

        if isinstance(o, number):

            object_range = max(objects) - min(objects)

            if object_range:
                o_01 = (o - min(objects)) / object_range
                i = int(o_01 * cmap.N)

            else:
                i = 0

        c = cmap(i)

        o_to_color[o] = c

    return o_to_color
