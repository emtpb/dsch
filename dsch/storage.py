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
import os

from . import frontend


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
        storage_path (str): Path to the current storage (backend-specific).
        schema_node: Top-level schema node used for the stored data.
        data: Top-level data node, providing access to all managed data.
    """

    def __init__(self, storage_path, schema_node=None):
        """Initialize the storage interface.

        To create a new storage file, ``storage_path`` and ``schema_node`` must
        be specified. When loading an existing file, ``schema_node`` must not
        be given.

        .. note::
            Most backends cache changes to the data in memory. For writing
            these to disk or commit them to a database (backend-specific), they
            provide a method like ``save`` that must be called explicitly.

        Args:
            storage_path (str): Path to the storage (backend-specific).
            schema_node: Top-level schema node for the data hierarchy.
        """
        self.data = None
        self.storage_path = storage_path
        self.schema_node = schema_node

    @property
    def complete(self):
        """Check whether the stored data is currently complete.

        The data held by this storage interface is considered complete when the
        top-level data node is complete. In most cases, this will be a
        :class:`dsch.data.Compilation` or :class:`dsch.data.List`, which
        recursively check their sub-nodes for completeness.
        When the Storage is complete, this means that all required data fields
        are filled out. Compilation fields marked as optional via
        :attr:`.schema.Compilation.optionals` are not considered in this
        process.

        Returns:
            bool: ``True`` if the stored data is complete, ``False`` otherwise.
        """
        return self.data.complete

    def schema_hash(self):
        """Calculate the SHA256 hash of the (serialized) schema.

        This uses the JSON-serialized schema specification that is mostly
        included with the respective dsch storage.

        Returns:
            str: SHA256 hash (hex) of the schema.
        """
        return self.schema_node.hash()

    def save_as(self, storage_path, backend=None):
        """Create a new storage by copying schema and data.

        .. note::
            Creating a copy can be useful to migrate existing data to a
            different storage backend, or to persistently store data that was
            collected in an in-memory storage.

        This is a convenience method, effectively wrapping
        :func:`~dsch.frontend.create_from`.

        Args:
            storage_path (str): Path to the new dsch storage
                (backend-specific).
            backend (str): Backend to use for the new dsch storage. If omitted,
                the backend will be selected based on the ``storage_path``,
                e.g. file extension.

        Returns:
            Newly created dsch storage.
        """
        return frontend.create_from(storage_path, self, backend)

    def validate(self):
        """Validate the entire data storage.

        This recursively validates all individual data nodes inside the
        storage.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            dsch.exceptions.ValidationError: if validation fails and the
                top-level node is a regular node.
            dsch.exceptions.SubnodeValidationError: if validation fails and the
                top-level node is a :class:`~dsch.data.Compilation` or
                :class:`~dsch.data.List`.
        """
        self.data.validate()


class FileStorage(Storage):
    """Storage interface base class for file-based storage.

    FileStorage extends :class:`Storage` by common functionality that is shared
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
        be specified. When loading an existing file, ``schema_node`` must not
        be given.

        .. note::
            File-based backends cache changes to the data in memory. For
            writing these to disk, they provide a :meth:`save` method that must
            be called explicitly.

        Args:
            storage_path (str): Path to the storage file.
            schema_node: Top-level schema node for the data hierarchy.

        Raises:
            FileExistsError: when trying to create a new storage with an
                existing path.
            FileNotFoundError: when trying to open a file that does not exist.
        """
        super().__init__(storage_path, schema_node)
        if schema_node:
            # If schema_node is given, we're creating a new file
            if os.path.exists(self.storage_path):
                raise FileExistsError('File %s already exists.',
                                      self.storage_path)
            self._new()
        else:
            if not os.path.exists(self.storage_path):
                raise FileNotFoundError('File %s could not be found.',
                                        self.storage_path)
            self._load()

    def _load(self):
        """Load an existing file from :attr:`storage_path`."""
        raise NotImplementedError('To be implemented in subclass.')

    def _new(self):
        """Create a new file at :attr:`storage_path`."""
        raise NotImplementedError('To be implemented in subclass.')

    def save(self, force=False):
        """Save the current data to the file in :attr:`storage_path`.

        Before the file is saved, data validation is automatically performed
        via :meth:`validate`. This can be skipped (although it should not) by
        setting ``force`` to ``True``.

        Args:
            force (bool): If ``True``, automatic data validation is skipped.
        """
        if not force:
            self.validate()
        self._save()

    def _save(self):
        """Save the current data to the file in :attr:`storage_path`."""
        raise NotImplementedError('To be implemented in subclass.')
