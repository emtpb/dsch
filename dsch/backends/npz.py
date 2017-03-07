"""dsch backend for NumPy's npz format.

This backend provides support for NumPy's npz format. For details, see
:func:`numpy.savez`, :func:`numpy.load` and the `corresponding NumPy
enhancement proposal <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_.
"""
import json
import numpy as np
from .. import data, helpers, schema


class Bool(data.ItemNode):
    """Bool-type data node for the npz backend."""

    def load(self, data_storage):
        """Import the given data storage object.

        Data storage depends on the current backend, so a compatible argument
        must be given.

        Args:
            data_storage: Data storage object to be imported.
        """
        self.storage = data_storage

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self.storage = np.array([new_value], dtype='bool')

    def save(self):
        """Export the node data as a data storage object.

        Returns:
            dict: Data storage object with the node's data.
        """
        return self.storage

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return bool(self.storage)


class Compilation(data.Compilation):
    """Compilation-type data node for the npz backend."""

    def load(self, data_storage):
        """Import the given data storage object.

        Data storage depends on the current backend, so a compatible argument
        must be given.

        Args:
            data_storage (dict): Data storage object to be imported.
        """
        for name, node in self._subnodes.items():
            if name in data_storage:
                node.load(data_storage[name])

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


class NPZFile:
    """Interface to ``.npz`` files.

    Provides access to an ``.npz`` file via dsch, i.e. reading from and writing
    to such a file.

    Attributes:
        schema_node: The top-level schema node for the file.
        data: The top-level data node, corresponding to :attr:`schema_node`.
    """

    def __init__(self, file_name=None, schema_node=None):
        """Initialize the ``.npz`` file interface.

        The interface can be initialized with either a path to a file which is
        then loaded, or with a top-level schema node to be applied so that a
        new (empty) data structure is created.

        Args:
            file_name (str): Path to the file to be loaded
            schema_node: Top-level schema node to set for the file.
        """
        if file_name and not schema_node:
            self.load(file_name)
        elif schema_node and not file_name:
            self.schema_node = schema_node
            self.data = data.data_node_from_schema(schema_node,
                                                   self.__module__)
        else:
            raise ValueError('Must initialize with either a file name or '
                             'a top-level schema node.')

    def load(self, file_name):
        """Load the contents of a given file into the :class:`NPZFile` object.

        Note: This sets :attr:`schema_node` according to the file, possibly
        overwriting a value previously set by the user.

        Args:
            file_name (str): Path to the file to be loaded.
        """
        with np.load(file_name) as file_:
            schema_data = json.loads(file_['_schema'][()])
            stored_data = helpers.inflate_dotted(file_)
        schema_node = schema.node_from_dict(schema_data)
        data_node = data.data_node_from_schema(schema_node, self.__module__)

        if isinstance(schema_node, schema.Compilation):
            del stored_data['_schema']
            data_node.load(stored_data)
        else:
            # If the top-level node is not a Compilation, the default name
            # 'data' is used for the node.
            data_node.load(stored_data['data'])
        self.data = data_node

    def save(self, file_name):
        """Save the current data to a file.

        Note: This does not perform any validation, so the created file is
        explicitly not guaranteed to fulfill the schema's constraints.

        Args:
            file_name (str): Path to the file to write to.
        """
        schema_str = json.dumps(self.schema_node.to_dict(), sort_keys=True)
        if isinstance(self.schema_node, schema.Compilation):
            store_data = helpers.flatten_dotted(self.data.save())
        else:
            # If the top-level node is not a Compilation, the default name
            # 'data' is used for the node.
            store_data = helpers.flatten_dotted({'data': self.data.save()})
        np.savez(file_name, _schema=schema_str, **store_data)


class List(data.List):
    """List-type data node for the npz backend."""

    def load(self, data_storage):
        """Import the given data storage object.

        Data storage depends on the current backend, so a compatible argument
        must be given.

        Args:
            data_storage (list): Data storage object to be imported.
        """
        for name, node_storage in sorted(data_storage.items()):
            node = data.data_node_from_schema(self.schema_node.subnode,
                                              self.__module__)
            node.load(node_storage)
            self._subnodes.append(node)

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


class String(data.ItemNode):
    """String-type data node for the npz backend."""

    def load(self, data_storage):
        """Import the given data storage object.

        Data storage depends on the current backend, so a compatible argument
        must be given.

        Args:
            data_storage: Data storage object to be imported.
        """
        self.storage = data_storage

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self.storage = np.array(new_value, dtype='U')

    def save(self):
        """Export the node data as a data storage object.

        Returns:
            dict: Data storage object with the node's data.
        """
        return self.storage

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return str(self.storage)
