from pandas import Series

from REVEALER.ccal.path import establish_path


def merge_dicts_with_function(dict_0, dict_1, function):
    """
    Merge dict_0 and dict_1, apply function to values keyed by the same key.
    Arguments:
        dict_0 (dict);
        dict_1 (dict):
        function (callable):
    Returns:
        dict: merged dict
    """

    merged_dict = {}

    for k in dict_0.keys() | dict_1.keys():

        if k in dict_0 and k in dict_1:
            merged_dict[k] = function(dict_0.get(k), dict_1.get(k))

        elif k in dict_0:
            merged_dict[k] = dict_0.get(k)

        elif k in dict_1:
            merged_dict[k] = dict_1.get(k)

        else:
            raise ValueError('dict_0 or dict_1 changed during iteration.')

    return merged_dict


def write_dict(dict_, file_path, key_name, value_name):
    """
    Write dict_ as .dict.tsv file.
    Arguments:
        dict_ (dict):
        file_path (str):
        key_name (str):
        value_name (str):
    Returns:
        None
    """

    s = Series(dict_, name=value_name)
    s.index.name = key_name

    if not file_path.endswith('.dict.tsv'):
        file_path += '.dict.tsv'

    establish_path(file_path)

    s.to_csv(file_path, sep='\t')
