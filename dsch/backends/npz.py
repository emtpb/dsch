"""dsch backend for NumPy's npz format.

This backend provides support for NumPy's npz format. For details, see
:func:`numpy.savez`, :func:`numpy.load` and the `corresponding NumPy
enhancement proposal <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_.
"""
import numpy as np
from .. import data, helpers, schema, storage


class Array(data.ItemNode):
    """Array-type data node for the npz backend."""

    def __getitem__(self, key):
        """Pass slicing/indexing operations directly to NumPy array."""
        return self._storage[key]

    def __setitem__(self, key, value):
        """Pass slicing/indexing operations directly to NumPy array."""
        self._storage[key] = value

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self._storage = new_value

    def resize(self, size):
        """Resize the array to the desired size.

        Args:
            size (tuple): Desired array shape.
        """
        self._storage.resize(size)

    def save(self):
        """Export the node data as a data storage object.

        Returns:
            dict: Data storage object with the node's data.
        """
        return self._storage

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return self._storage


class Bool(data.ItemNode):
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

    def save(self):
        """Export the node data as a data storage object.

        Returns:
            dict: Data storage object with the node's data.
        """
        return self._storage

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

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
        for name, node in self._subnodes.items():
            data_storage[name] = node.save()
        return data_storage


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


class Storage(storage.FileStorage):
    """Interface to ``.npz`` files.

    Provides access to an ``.npz`` file via dsch, i.e. reading from and writing
    to such a file.

    Attributes:
        storage_path (str): Path to the current storage.
        schema_node: Top-level schema node used for the stored data.
        data: Top-level data node, providing access to all managed data.
    """

    def _new(self):
        """Create a new file at :attr:`storage_path`."""
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__)

    def _load(self):
        """Load an existing file from :attr:`storage_path`."""
        with np.load(self.storage_path) as file_:
            self._schema_from_json(file_['_schema'][()])
            stored_data = helpers.inflate_dotted(file_)

        if isinstance(self.schema_node, schema.Compilation):
            data_storage = {k: v for k, v in stored_data.items()
                            if k != '_schema'}
        else:
            # If the top-level node is not a Compilation, the default name
            # 'data' is used for the node.
            data_storage = stored_data['data']
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__,
                                               data_storage=data_storage)

    def save(self):
        """Save the current data to the file in :attr:`storage_path`.

        Note: This does not perform any validation, so the created file is
        *not* guaranteed to fulfill the schema's constraints.
        """
        if isinstance(self.schema_node, schema.Compilation):
            store_data = helpers.flatten_dotted(self.data.save())
        else:
            # If the top-level node is not a Compilation, the default name
            # 'data' is used for the node.
            store_data = helpers.flatten_dotted({'data': self.data.save()})
        np.savez(self.storage_path, _schema=self._schema_to_json(),
                 **store_data)


class String(data.ItemNode):
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

    def save(self):
        """Export the node data as a data storage object.

        Returns:
            dict: Data storage object with the node's data.
        """
        return self._storage

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return str(self._storage)
