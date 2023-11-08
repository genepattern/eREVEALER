from numpy import apply_along_axis, isnan
from scipy.stats import rankdata

from REVEALER.ccal.normalize_1d_array import normalize_1d_array


def normalize_2d_array(array_2d, method, axis=None, rank_method='average'):
    """
    Normalize array_2d.
    Arguments:
        array_2d (array): (n, m)
        method (str): '-0-' | '0-1' | 'rank'
        axis (int | str): 'global' | 0 | 1 |
        rank_method (str): 'average' | 'min' | 'max' | 'dense' | 'ordinal'
    Returns:
        array: (n, m)
    """

    if axis is None:

        values = array_2d[~isnan(array_2d)]
        size = values.size
        mean = values.mean()
        std = values.std()
        min_ = values.min()
        max_ = values.max()

        if method == '-0-':

            if std:
                return (array_2d - mean) / std
            else:
                #print('std == 0: / size instead of 0-1 ...')
                return array_2d / size

        elif method == '0-1':

            if max_ - min_:
                return (array_2d - min_) / (max_ - min_)
            else:
                #print('(max - min) ==  0: / size instead of 0-1 ...')
                return array_2d / size

        elif method == 'rank':

            array_2d[isnan(array_2d)] = mean

            return (rankdata(array_2d, method=rank_method) /
                    size).reshape(array_2d.shape)

        else:
            raise ValueError('Unknown method: {}.'.format(method))

    elif axis == 0 or axis == 1:

        return apply_along_axis(
            normalize_1d_array,
            axis,
            array_2d,
            method=method,
            rank_method=rank_method)

    else:
        raise ValueError('Unknown axis: {}.'.format(axis))
