"""Helper module for functions not directly tied to the project structure."""


def inflate_dotted(input):
    """Convert flat dict with dotted key notation into nested dict.

    Given a flat dict with dotted keys for nested items, e.g.
        >>> input = {'spam.eggs': 23, 'answer': 42}

    create a nested dict
        >>> output = inflate_dotted(input)
        >>> output == {'spam': {'eggs': 23}, 'answer': 42}
        True

    Args:
        dict: Flat dict with dotted key notation.

    Returns:
        dict: Nested dict.
    """
    output = {}
    for key, value in input.items():
        ref = output
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in ref:
                ref[part] = {}
            ref = ref[part]
        ref[parts[-1]] = value
    return output


def flatten_dotted(input, prefix=''):
    """Convert nested dict into flat dict with dotted key notation.

    Given a nested dict, e.g.
        >>> input = {'spam': {'eggs': 23}, 'answer': 42}

    create a flat dict with dotted keys for the nested items
        >>> output = flatten_dotted(input)
        >>> output == {'spam.eggs': 23, 'answer': 42}
        True

    Args:
        dict: Nested dict
        string: Key prefix, used for recursion. Should be left at the default
            for normal use.

    Returns:
        dict: Flattened dict with dotted notation.
    """
    output = {}
    for key, value in input.items():
        if prefix:
            dotted = prefix + '.' + key
        else:
            dotted = key
        if isinstance(value, dict):
            output.update(flatten_dotted(value, dotted))
        else:
            output[dotted] = value
    return output
