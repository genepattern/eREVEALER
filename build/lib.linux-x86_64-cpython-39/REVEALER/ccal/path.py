from os import mkdir, remove
from os.path import abspath, exists, expanduser, isdir, islink, split
from shutil import copy, copytree, rmtree


def establish_path(path, path_type='file'):
    """
    Make directory paths up to the deepest directory in path.
    Arguments:
        path (str):
        path_type (str): 'file' | 'directory'
    Returns:
        None
    """

    # Work with absolute path
    path = abspath(path)

    # Check path ending
    if path_type == 'file':
        if path.endswith('/'):
            raise ValueError('File path should not end with \'/\'.')
    else:
        if not path.endswith('/'):
            path += '/'

    directory_path, file_name = split(path)

    # List missing directory paths
    missing_directory_paths = []

    while not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        # Check directory_path's directory_path
        directory_path, file_name = split(directory_path)

    # Make missing directories
    for directory_path in reversed(missing_directory_paths):
        mkdir(directory_path)
        print('Created directory {}.'.format(directory_path))


def copy_path(from_path, to_path, overwrite=False):
    """
    Copy from_path to to_path.
    Arguments:
        from_path (str):
        to_path (str):
        overwrite (bool):
    Returns:
        None
    """

    if overwrite:
        remove_path(to_path)

    if isdir(from_path):
        copytree(from_path, to_path)

    elif exists(from_path):
        copy(from_path, to_path)


def remove_path(path):
    """
    Remove path.
    Arguments:
        path (str):
    Returns:
        None
    """

    if islink(path):
        remove(path)

    elif isdir(path):
        rmtree(path)

    elif exists(path):
        remove(path)


def clean_path(path):
    """
    Clean path.
    Argument:
        path (str):
    Returns:
        str:
    """

    return abspath(expanduser(path))


def is_good_name(name):
    """
    Check if name is good.
    Arguments:
        name (str):
    Returns:
        bool:
    """

    if '/' in name:
        return False

    else:
        return True
