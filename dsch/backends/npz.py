"""dsch backend for NumPy's npz format.

This backend provides support for NumPy's npz format. For details, see
:func:`numpy.savez`, :func:`numpy.load` and the `corresponding NumPy
enhancement proposal <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_.
"""
import numpy as np
from .. import data


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
