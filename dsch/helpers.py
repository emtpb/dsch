"""Helper module for functions not directly tied to the project structure."""
import importlib


def backend_module(backend_name):
    """Return the backend module corresponding to the given name.

    Note: The module is automatically imported if it is accessed for the first
    time.

    Args:
        backend_name (str): Name of the backend module to find.

    Returns:
        module: Backend module for the given name.
    """
    return importlib.import_module('dsch.backends.' + backend_name)


def inflate_dotted(input_dict):
    """Convert flat dict with dotted key notation into nested dict.

    Given a flat dict with dotted keys for nested items, e.g.
        >>> input_dict = {'spam.eggs': 23, 'answer': 42}

    create a nested dict
        >>> output_dict = inflate_dotted(input_dict)
        >>> output_dict == {'spam': {'eggs': 23}, 'answer': 42}
        True

    Args:
        dict: Flat dict with dotted key notation.

    Returns:
        dict: Nested dict.
    """
    output_dict = {}
    for key, value in input_dict.items():
        ref = output_dict
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in ref:
                ref[part] = {}
            ref = ref[part]
        ref[parts[-1]] = value
    return output_dict


def flatten_dotted(input_dict, prefix=''):
    """Convert nested dict into flat dict with dotted key notation.

    Given a nested dict, e.g.
        >>> input_dict = {'spam': {'eggs': 23}, 'answer': 42}

    create a flat dict with dotted keys for the nested items
        >>> output_dict = flatten_dotted(input_dict)
        >>> output_dict == {'spam.eggs': 23, 'answer': 42}
        True

    Args:
        dict: Nested dict
        string: Key prefix, used for recursion. Should be left at the default
            for normal use.

    Returns:
        dict: Flattened dict with dotted notation.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if prefix:
            dotted = prefix + '.' + key
        else:
            dotted = key
        if isinstance(value, dict):
            output_dict.update(flatten_dotted(value, dotted))
        else:
            output_dict[dotted] = value
    return output_dict
