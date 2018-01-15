"""dsch backend for in-memory data storage.

This backend stores all data in memory and cannot directly save to disk.
For temporary data that does not have to be stored, the in-memory backend
provides a clean way of data with dsch, without littering the workspace with
temporary files. Also, it can be used to collect and aggregate data *before*
selecting a storage path, e.g. file name.
"""
from .. import data, storage


class _ItemNode(data.ItemNode):
    """Common base class for data nodes for the inmem backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        # Directly store the given value, since there is no storage engine that
        # might require any data type changes etc.
        self._storage = new_value

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        # Directly return the given value, since there is no storage engine
        # that might require any data type changes etc.
        return self._storage


class Array(data.Array, _ItemNode):
    """Array-type data node for the inmem backend."""
    pass


class Bytes(_ItemNode):
    """Bytes-type data node for the inmem backend."""
    pass


class Bool(_ItemNode):
    """Bool-type data node for the inmem backend."""
    pass


class Compilation(data.Compilation):
    """Compilation-type data node for the inmem backend."""
    pass


class Date(data.Date, _ItemNode):
    """Date-type data node for the inmem backend."""
    pass


class DateTime(data.DateTime, _ItemNode):
    """DateTime-type data node for the inmem backend."""
    pass


class List(data.List):
    """List-type data node for the inmem backend."""
    pass


class Scalar(_ItemNode):
    """Scalar-type data node for the inmem backend."""
    pass


class Storage(storage.Storage):
    """Interface to the in-memory storage.

    Attributes:
        storage_path (str): Path to the current storage.
        schema_node: Top-level schema node used for the stored data.
        data: Top-level data node, providing access to all managed data.
    """

    def __init__(self, storage_path='::inmem::', schema_node=None):
        """Initialize the in-memory storage interface.

        .. note::
            The in-memory backend does not support saving data to disk, so no
            storage path must be given. Consequently, it does not support
            loading data, so the only possible operation is creating a new,
            empty storage.

        Args:
            storage_path: Only supported for API compatibility. If given, this
                must always be "::inmem::".
            schema_node: Top-level schema node for the data hierarchy.
        """
        if storage_path != '::inmem::':
            raise ValueError('Invalid storage path for in-memory backend. '
                             'Must be the special string "::inmem::".')
        if not schema_node:
            raise ValueError('Top-level schema node must always be specified.')
        super().__init__(storage_path, schema_node)
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None)


class String(_ItemNode):
    """String-type data node for the inmem backend."""
    pass


class Time(data.Time, _ItemNode):
    """Time-type data node for the inmem backend."""
    pass
