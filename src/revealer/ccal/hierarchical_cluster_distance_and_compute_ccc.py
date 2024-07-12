from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


def hierarchical_cluster_distance_and_compute_ccc(distance__sample_x_sample,
                                                  zero_self_distance=True,
                                                  method='ward'):
    """
    Hierarchical cluster distance__sample_x_sample and compute cophenetic
        correlation coefficient (CCC).
    Arguments:
        distance__sample_x_sample (array): (n_sample, n_sample)
        zero_self_distance (bool): whether to force self distance
            (distance[i, i]) to be 0
        method (str): method parameter compatible for
            scipy.cluster.hierarchy.linkage
    Returns:
        array: z (linkage)
        float: CCC
    """

    if zero_self_distance:
        for i in range(distance__sample_x_sample.shape[0]):
            distance__sample_x_sample[i, i] = 0

    hc = linkage(distance__sample_x_sample, method=method)

    return hc, pearsonr(cophenet(hc), pdist(distance__sample_x_sample))[0]
