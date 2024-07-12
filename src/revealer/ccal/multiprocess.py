from multiprocessing.pool import Pool

from numpy.random import seed


def multiprocess(function, args, n_jobs, random_seed=None):
    """
    Call function with args across n_jobs processes (n_jobs doesn't have to be
        the length of list_of_args).
    Arguments:
        function (callable):
        args (iterable): an iterable of [(1,2), (3, 4)] results in
            [function(1,2), function(3,4)]
        n_jobs (int): 0 < n_jobs
        random_seed (int | array):
    Returns:
        list:
    """

    if random_seed is not None:
        # Each process initializes with the current jobs' randomness (random
        # state & random state index). Any changes to these processes'
        # randomnesses won't update the current process' randomness.
        seed(random_seed)

    with Pool(n_jobs) as p:
        return p.starmap(function, args)
