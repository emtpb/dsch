"""dsch storage representation.

The data node classes provided by :mod:`dsch.data` form the abstraction layer
between the different backend's specific data storage mechanisms for the
individual node types (i.e. data types) modeled by dsch.
In addition to that, the storage entity itself, e.g. a file or database, must
be made available to the user, consequently using data nodes to model the data
fields. The structure and hierarchy of these nodes is determined by the schema,
using the classes from :mod:`dsch.schema`.

This module provides base classes for backends to derive from, so that common
functionality may be implemented in a single place without unnecessary
repetition.
"""
import json
import os
from . import schema


class Storage:
    """Generic storage interface base class.

    Storage interfaces provide access to a specific data storage that is
    managed by dsch. Depending on the specific backend, this can for example be
    a file, a directory or a database.

    Once created, the :class:`Storage` provides access to all contained data
    via :attr:`data`. Internally, this maps to the top-level data node in the
    hierarchy.

    .. warning::
        Once created, changes to :attr:`schema_node` are not automatically
        propagated through the data node tree, so no changes should be made
        to it while using a :class:`Storage` object.

    Attributes:
        storage_path (str): Path to the current storage.
        schema_node: Top-level schema node used for the stored data.
        data: Top-level data node, providing access to all managed data.
    """

    def __init__(self, storage_path, schema_node=None):
        """Initialize the storage interface.

        To create a new storage, ``storage_path`` and ``schema_node`` must be
        specified. Note that most backends do not automatically write data
        changes to disk until :meth:`save` is called.

        To open a storage that already exists, only ``storage_path`` must be
        specified. In this case, ``schema_node`` is ignored, if given
        additionally.

        Args:
            storage_path (str): Path to the storage (format depending on the
                specific backend).
            schema_node: Top-level schema node for the data hierarchy.
        """
        self.storage_path = storage_path
        self.schema_node = schema_node

    def _schema_from_json(self, json_str):
        """Import the top-level schema node from a JSON string.

        Imports the given JSON string and creates a corresponding schema node
        in :attr:`schema_node`.

        Args:
            json_str (str): JSON string representing the schema node.
        """
        self.schema_node = schema.node_from_dict(json.loads(json_str))

    def _schema_to_json(self):
        """Export the top-level schema node as a JSON string.

        Returns:
            str: JSON representation of :attr:`schema_node`
        """
        return json.dumps(self.schema_node.to_dict(), sort_keys=True)


class FileStorage(Storage):
    """Storage interface base class for file-based storage.

    FileStorage expand :class:`Storage` by common functionality that is shared
    by all file-based storage mechanisms. This also provides a common interface
    to the user, independent of the specific file format (i.e. backend) in use.

    Attributes:
        storage_path (str): Path to the current storage file.
        schema_node: Top-level schema node used for the stored data.
        data: Top-level data node, providing access to all managed data.
    """

    def __init__(self, storage_path, schema_node=None):
        """Initialize the storage interface to a file.

        To create a new storage file, ``storage_path`` and ``schema_node`` must
        be specified. Note that most backends do not automatically write data
        changes to disk until :meth:`save` is called.

        To open a storage file that already exists, only ``storage_path`` must
        be specified. In this case, ``schema_node`` is ignored, if given
        additionally.

        Args:
            storage_path (str): Path to the storage (format depending on the
                specific backend).
            schema_node: Top-level schema node for the data hierarchy.
        """
        super().__init__(storage_path, schema_node)
        if os.path.exists(self.storage_path):
            self._load()
        else:
            self._new()

    def _load(self):
        """Load an existing file from :attr:`storage_path`."""
        raise NotImplementedError('To be implemented in subclass.')

    def _new(self):
        """Create a new file at :attr:`storage_path`."""
        raise NotImplementedError('To be implemented in subclass.')

    def save(self):
        """Save the current data to the file in :attr:`storage_path`.

        Note: This does not perform any validation, so the created file is
        *not* guaranteed to fulfill the schema's constraints.
        """
        raise NotImplementedError('To be implemented in subclass.')
