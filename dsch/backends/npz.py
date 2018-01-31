"""dsch backend for NumPy's npz format.

This backend provides support for NumPy's npz format. For details, see
:func:`numpy.savez`, :func:`numpy.load` and the `corresponding NumPy
enhancement proposal <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_.
"""
import datetime

import numpy as np

from .. import data, schema, storage


class _ItemNode(data.ItemNode):
    """Common base class for data nodes for the npz backend."""

    def save(self):
        """Export the node data as a data storage object.

        Returns:
            dict: Data storage object with the node's data.
        """
        return self._storage


class Array(data.Array, _ItemNode):
    """Array-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = new_value

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return self._storage


class Bytes(_ItemNode):
    """Bytes-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = np.array(new_value, dtype='bytes')

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return bytes(self._storage)


class Bool(_ItemNode):
    """Bool-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = np.array([new_value], dtype='bool')

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return bool(self._storage)


class Compilation(data.Compilation):
    """Compilation-type data node for the npz backend."""

    def save(self):
        """Export the node data as a data storage object.

        For :class:`Compilation`, data is represented as a :class:`dict`
        containing the sub-node names as keys and their respective data storage
        object as values.

        Returns:
            dict: Data storage object with the node's data.
        """
        data_storage = {}
        for name, node in [(na, no) for na, no in self._subnodes.items()
                           if not no.empty]:
            data_storage[name] = node.save()
        return data_storage


class Date(data.Date, _ItemNode):
    """Date-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = np.array([new_value.year, new_value.month,
                                  new_value.day], dtype='int')

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return datetime.date(*self._storage.tolist())


class DateTime(data.DateTime, _ItemNode):
    """DateTime-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = np.array([new_value.year, new_value.month,
                                  new_value.day, new_value.hour,
                                  new_value.minute, new_value.second,
                                  new_value.microsecond], dtype='int')

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return datetime.datetime(*self._storage.tolist())


class List(data.List):
    """List-type data node for the npz backend."""

    def save(self):
        """Export the node data as a data storage object.

        For :class:`List`, data is represented as a :class:`dict` containing
        the sub-node's data storage objects as values and keys of the form
        ``item_X``, where ``X`` is the list index.

        Returns:
            dict: Data storage object with the node's data.
        """
        data_storage = {}
        for idx, node in enumerate(self._subnodes):
            data_storage['item_{}'.format(idx)] = node.save()
        return data_storage


class Scalar(data.Scalar, _ItemNode):
    """Scalar-type data node for the npz backend."""

    def _init_from_storage(self, data_storage):
        """Create a new data node from a data storage object.

        This initializes the data node using the given data storage object.

        Args:
            data_storage: Backend-specific data storage object to load.
        """
        self._storage = np.dtype(self.schema_node.dtype).type(data_storage)

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = np.dtype(self.schema_node.dtype).type(new_value)

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return self._storage


class Storage(storage.FileStorage):
    """Interface to ``.npz`` files.

    Provides access to an ``.npz`` file via dsch, i.e. reading from and writing
    to such a file.

    Attributes:
        storage_path (str): Path to the current storage.
        schema_node: Top-level schema node used for the stored data.
        data: Top-level data node, providing access to all managed data.
    """

    def _load(self):
        """Load an existing file from :attr:`storage_path`."""
        with np.load(self.storage_path) as file_:
            self.schema_node = schema.node_from_json(file_['_schema'][()])
            stored_data = _inflate_dotted(file_)

        if isinstance(self.schema_node, schema.Compilation):
            data_storage = {k: v for k, v in stored_data.items()
                            if k != '_schema'}
        else:
            # If the top-level node is not a Compilation, the default name
            # 'data' is used for the node.
            data_storage = stored_data.get('data', None)
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None,
                                               data_storage=data_storage)

    def _new(self):
        """Create a new file at :attr:`storage_path`."""
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None)

    def _save(self):
        """Save the current data to the file in :attr:`storage_path`."""
        if isinstance(self.schema_node, schema.Compilation):
            store_data = _flatten_dotted(self.data.save())
        else:
            # If the top-level node is not a Compilation, the default name
            # 'data' is used for the node.
            output_data = self.data.save()
            if output_data is not None:
                store_data = _flatten_dotted({'data': output_data})
            else:
                store_data = {}
        np.savez(self.storage_path, _schema=self.schema_node.to_json(),
                 **store_data)


class String(_ItemNode):
    """String-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = np.array(new_value, dtype='U')

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return str(self._storage)


class Time(data.Time, _ItemNode):
    """Time-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = np.array([new_value.hour, new_value.minute,
                                  new_value.second, new_value.microsecond],
                                 dtype='int')

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return datetime.time(*self._storage.tolist())


def _inflate_dotted(input_dict):
    """Convert flat dict with dotted key notation into nested dict.

    Given a flat dict with dotted keys for nested items, e.g.
        >>> input_dict = {'spam.eggs': 23, 'answer': 42}

    create a nested dict
        >>> output_dict = _inflate_dotted(input_dict)
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


def _flatten_dotted(input_dict, prefix=''):
    """Convert nested dict into flat dict with dotted key notation.

    Given a nested dict, e.g.
        >>> input_dict = {'spam': {'eggs': 23}, 'answer': 42}

    create a flat dict with dotted keys for the nested items
        >>> output_dict = _flatten_dotted(input_dict)
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
            output_dict.update(_flatten_dotted(value, dotted))
        else:
            output_dict[dotted] = value
    return output_dict
