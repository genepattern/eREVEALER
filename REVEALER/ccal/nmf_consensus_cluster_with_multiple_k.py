from os.path import join

from pandas import DataFrame, Series

from REVEALER.ccal.nmf_consensus_cluster import nmf_consensus_cluster
from REVEALER.ccal.plot_heatmap import plot_heatmap
from REVEALER.ccal.plot_nmf import plot_nmf
from REVEALER.ccal.plot_points import plot_points
from REVEALER.ccal.multiprocess import multiprocess
from REVEALER.ccal.path import establish_path

RANDOM_SEED = 20121020


def nmf_consensus_cluster_with_multiple_k(df,
                                          ks,
                                          n_jobs=1,
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
                                          directory_path=None):
    """
    NMF consensus cluster (NMFCC) df columns with multiple ks and compute
        cophenetic correlation coefficients (CCCs).
    Arguments:
        df (DataFrame): (n_feature, n_sample)
        ks (iterable): (n_k)
        n_jobs (int):
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
        directory_path (str):
    Returns:
        dict:  k_nmf;
            {
                k: {
                    w: W matrix,
                    h: H matrix,
                    e: reconstruction error
                }
            }
        DataFrame: (n_k, n_sample); nmfcc__k_x_sample
        Series: (n_k); nmfcc__k_ccc
    """

    if directory_path:
        establish_path(directory_path, path_type='directory')

    k_nmf = {}
    indices = ['K{}'.format(k) for k in ks]
    nmfcc__k_x_sample = DataFrame(index=indices, columns=df.columns)
    nmfcc__k_ccc = Series(index=indices, name='CCC')

    args = [(df, k, n_clustering, algorithm, init, solver, beta_loss, tol,
             max_iter, random_seed, alpha, l1_ratio, verbose, shuffle, False,
             directory_path) for k in ks]
    returns = multiprocess(nmf_consensus_cluster, args, n_jobs=n_jobs)

    for k, (w, h, e, nmfcc__clusters, ccc) in zip(ks, returns):

        k_key = 'K{}'.format(k)
        k_nmf[k_key] = {'w': w, 'h': h, 'e': e}
        nmfcc__k_x_sample.loc[k_key] = nmfcc__clusters
        nmfcc__k_ccc[k_key] = ccc

        if directory_path:
            p = join(directory_path, 'nmf_k{}'.format(k))
        else:
            p = None
        plot_nmf(
            k_nmf['K{}'.format(k)]['w'],
            k_nmf['K{}'.format(k)]['h'],
            max_std=3,
            file_path_prefix=p)

        if directory_path:
            p = join(directory_path, 'nmfcc_k{}.png'.format(k))
        else:
            p = None
        plot_heatmap(
            df,
            column_annotation=nmfcc__clusters,
            normalization_method='-0-',
            normalization_axis=1,
            title='NMFCC with K={}'.format(k),
            file_path=p)

    if directory_path:
        nmfcc__k_x_sample.to_csv(
            join(directory_path, 'nmfcc__k_x_sample.tsv'), sep='\t')

    if directory_path:
        p = join(directory_path, 'nmfcc__k_x_sample__distribution.png')
    else:
        p = None
    plot_heatmap(
        nmfcc__k_x_sample,
        axis_to_sort=1,
        data_type='categorical',
        title='NMFCC Distribution',
        file_path=p)

    if directory_path:
        p = join(directory_path, 'nmfcc__k_ccc.png')
    else:
        p = None
    plot_points(
        sorted(ks),
        nmfcc__k_ccc,
        linestyle='-',
        markersize=8,
        title='NMFCC CCC',
        xlabel='K',
        ylabel='CCC',
        file_path=p)

    return k_nmf, nmfcc__k_x_sample, nmfcc__k_ccc
