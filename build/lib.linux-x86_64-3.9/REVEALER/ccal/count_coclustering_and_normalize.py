from numpy import zeros


def count_coclustering_and_normalize(sample_x_clustering):
    """
    Count the number of times samples are clustered together, and normalize the
        count by dividing it with the number of clusterings.
    Arguments:
        sample_x_clustering (array): (n_sample, n_clustering)
    Returns:
        array: (n_sample, n_sample); normalized_coclustering__sample_x_sample
    """

    n_coclustering__sample_x_sample = zeros([sample_x_clustering.shape[0]] * 2)

    for i in range(n_coclustering__sample_x_sample.shape[0]):
        for j in range(n_coclustering__sample_x_sample.shape[1]):
            for c_i in range(sample_x_clustering.shape[1]):

                v1 = sample_x_clustering[i, c_i]
                v2 = sample_x_clustering[j, c_i]

                if v1 and v2 and (v1 == v2):
                    n_coclustering__sample_x_sample[i, j] += 1

    return n_coclustering__sample_x_sample / sample_x_clustering.shape[1]
