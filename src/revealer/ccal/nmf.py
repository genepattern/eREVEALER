from numpy import divide, dot, finfo, log, matrix, multiply, ndarray
from numpy.random import rand, seed
from sklearn.decomposition import NMF

RANDOM_SEED = 20121020


def nmf(a,
        k,
        algorithm='als',
        init=None,
        solver='cd',
        beta_loss='frobenius',
        tol=0.0001,
        max_iter=200,
        random_seed=RANDOM_SEED,
        alpha=0.0,
        l1_ratio=0.0,
        verbose=0,
        shuffle=False):
    """
    Non-negative matrix factorize a using k: A ~ W * H.
    Arguments:
        a (array): (n_samples, n_features); A matrix
        k (int): number of hidden variables
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
    Returns:
        array: W matrix
        array: H matrix
        float: reconstruction error
    """

    if algorithm == 'als':

        model = NMF(
            n_components=k,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_seed,
            alpha=alpha,
            l1_ratio=l1_ratio,
            verbose=verbose,
            shuffle=shuffle)

        w = model.fit_transform(a)
        h = model.components_
        e = model.reconstruction_err_

    elif algorithm == 'ls':

        w, h, e = _nmf_div(
            a, k, n_iterations=max_iter, random_seed=random_seed)

    else:
        raise ValueError('Unknown algorithm: {}.'.format(algorithm))

    return w, h, e


def _nmf_div(a, k, n_iterations=1000, random_seed=RANDOM_SEED):
    """
    Non-negative matrix factorize matrix using k with divergence algorithm.
    Arguments:
        a (array): (n_samples, n_features); A matrix
        k (int): number of hidden variables
        n_max_iterations (int):
        random_seed (int | array):
    Returns:
        array: W matrix
        array: H matrix
        float: reconstruction error
    """

    # TODO: optimize

    eps = finfo(float).eps

    n = a.shape[0]
    m = a.shape[1]
    a = matrix(a)

    seed(random_seed)
    w = rand(n, k)
    h = rand(k, m)

    for t in range(n_iterations):

        a_p = dot(w, h)

        w_t = matrix.transpose(w)

        h = multiply(h, dot(w_t, divide(a, a_p))) + eps

        for i in range(k):
            w_sum = 0
            for j in range(n):
                w_sum += w[j, i]
            for j in range(m):
                h[i, j] = h[i, j] / w_sum

        a_p = dot(w, h)

        h_t = matrix.transpose(h)

        w = multiply(w, dot(divide(a, a_p + eps), h_t)) + eps

        w = divide(w, ndarray.sum(h, axis=1, keepdims=False))

    e = (multiply(a, log(divide(a + eps, a_p + eps))) - a + a_p).sum() / (
        m * n)

    return w, h, e
