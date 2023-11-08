from numpy import isnan
from scipy.stats import rankdata


def normalize_1d_array(array_1d, method, rank_method='average'):
    """
    Normalize array_1d.
    Arguments:
        array_1d (array): (n)
        method (str): '-0-' | '0-1' | 'rank'
        rank_method (str): 'average' | 'min' | 'max' | 'dense' | 'ordinal'
    Returns:
        array: (n)
    """

    values = array_1d[~isnan(array_1d)]
    size = values.size
    mean = values.mean()
    std = values.std()
    min_ = values.min()
    max_ = values.max()

    if method == '-0-':

        if std:
            return (array_1d - mean) / std
        else:
            #print('std == 0: / size instead of 0-1 ...')
            return array_1d / size

    elif method == '0-1':

        if max_ - min_:
            return (array_1d - min_) / (max_ - min_)
        else:
            #print('(max - min) ==  0: / size instead of 0-1 ...')
            return array_1d / size

    elif method == 'rank':

        # Assign mean to nans
        array_1d[isnan(array_1d)] = mean
        return rankdata(array_1d, method=rank_method) / size

    else:
        raise ValueError('Unknown method: {}.'.format(method))
