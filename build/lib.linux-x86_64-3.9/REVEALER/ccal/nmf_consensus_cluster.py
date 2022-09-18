from os.path import join

from numpy import argmax, empty
from pandas import DataFrame
from scipy.cluster.hierarchy import fcluster

from REVEALER.ccal.count_coclustering_and_normalize import count_coclustering_and_normalize
from REVEALER.ccal.hierarchical_cluster_distance_and_compute_ccc import \
    hierarchical_cluster_distance_and_compute_ccc
from REVEALER.ccal.nmf import nmf
from REVEALER.ccal.plot_heatmap import plot_heatmap
from REVEALER.ccal.plot_nmf import plot_nmf
from REVEALER.ccal.path import establish_path

RANDOM_SEED = 20121020


def nmf_consensus_cluster(df,
                          k,
                          n_clustering=30,
                          algorithm='als',
                          init=None,
                          solver='cd',
                          beta_loss='frobenius',
                          tol=0.0001,
                          max_iter=1000,
                          random_seed=RANDOM_SEED,
                          alpha=0.0,
                          l1_ratio=0.0,
                          verbose=0,
                          shuffle=False,
                          plot=True,
                          directory_path=None):
    """
    NMF consensus cluster (NMFCC) df columns with k and compute cophenetic
        correlation coefficient (CCC).
    Arguments:
        df (DataFrame): (n_feature, n_sample)
        k (int):
        n_clustering (int):
        algorithm (str): 'als' | 'ls'
        init:
        solver:
        beta_loss:
        tol:
        max_iter:
        random_seed (int | array):
        alpha:
        l1_ratio:
        verbose:
        shuffle:
        plot (bool):
        directory_path (str):
    Returns:
        array: (n, k); W matrix
        array: (k, m); H matrix
        float: reconstruction error
        array: (n_sample); nmfcc_clusters
        float: CCC
    """

    if directory_path:
        establish_path(directory_path, path_type='directory')

    if verbose != 0:
        print('NMFCC with K={} ...'.format(k))

    sample_x_clustering = empty((df.shape[1], n_clustering))

    for i in range(n_clustering):
        if i % 10 == 0 and verbose != 0:
            print('\t(K={}) {}/{} ...'.format(k, i + 1, n_clustering))

        w, h, e = nmf(
            df,
            k,
            algorithm=algorithm,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_seed=random_seed + i,
            alpha=alpha,
            l1_ratio=l1_ratio,
            verbose=verbose,
            shuffle=shuffle)

        if i == 0:
            w0 = w
            h0 = h
            e0 = e

            indices = ['C{}'.format(i) for i in range(k)]
            w0 = DataFrame(w0, index=df.index, columns=indices)
            h0 = DataFrame(h0, index=indices, columns=df.columns)

            if directory_path:
                w0.to_csv(
                    join(directory_path, 'nmf_k{}_w.tsv'.format(k)), sep='\t')
                h0.to_csv(
                    join(directory_path, 'nmf_k{}_h.tsv'.format(k)), sep='\t')

            if plot:
                if directory_path:
                    p = join(directory_path, 'nmf_k{}'.format(k))
                else:
                    p = None
                plot_nmf(w0, h0, max_std=3, file_path_prefix=p)

        sample_x_clustering[:, i] = argmax(h, axis=0)
    if verbose != 0:
        print('\t(K={}) {}/{} - done.'.format(k, i + 1, n_clustering))

    hc, ccc = hierarchical_cluster_distance_and_compute_ccc(
        1 - count_coclustering_and_normalize(sample_x_clustering))

    nmfcc_clusters = fcluster(hc, k, criterion='maxclust') - 1

    if plot:
        if directory_path:
            p = join(directory_path, 'nmfcc_k{}.png'.format(k))
        else:
            p = None
        plot_heatmap(
            df,
            column_annotation=nmfcc_clusters,
            normalization_method='-0-',
            normalization_axis=1,
            title='NMFCC with K={}'.format(k),
            file_path=p)

    return w0, h0, e0, nmfcc_clusters, ccc
