from numpy import array
from scipy.cluster.hierarchy import dendrogram, linkage


def cluster_2d_array_rows(array_2d,
                          linkage_method='average',
                          distance_function='euclidean'):
    """
    Cluster array_2d rows.
    Arguments:
        array_2d (array): (n_rows, n_columns)
        linkage_method (str): linkage method compatible for
            scipy.cluster.hierarchy.linkage
        distance_function (str | callable): distance function compatible for
            scipy.cluster.hierarchy.linkage
    Returns:
        array: (n_rows); clustered row indices
    """

    clustered_indices = dendrogram(
        linkage(array_2d, method=linkage_method, metric=distance_function),
        no_plot=True)['leaves']

    return array(clustered_indices)
