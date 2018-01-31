"""dsch user frontend.

When using dsch, users normally start with a top-level object representing
the dsch storage (e.g. a file) and work through that object's attributes.
Although they of course work with a variety of different objects in the
process, they usually do not create any of these manually.
When creating a *new* storage, a schema specification is required, which can be
built with the classes from :mod:`dsch.schema`. Then, a dsch storage can be
created.

The user front end in this module provides a convenient, backend-independent
interface for loading from existing dsch storages and creating new ones.
"""
import importlib
import os

from . import exceptions, schema


def create(storage_path, schema_node, backend=None):
    """Create a new dsch storage.

    Creates a new dsch storage in the location given by ``storage_path``, using
    the desired ``backend``. If no backend is specified, it is detected
    automatically by interpreting the ``storage_path``, e.g. via a file
    extension.
    Note that the format of ``storage_path`` depends on the choice of backends,
    so a compatible one must be chosen.

    Currently, the following backends are available:

    ==========  ==================  ========================
    Name        Description         Path format
    ==========  ==================  ========================
    hdf5        HDF5 file           Path to regular file
    inmem       In-memory storage   Fixed string "::inmem::"
    mat         MATLAB data file    Path to regular file
    npz         NumPy .npz file     Path to regular file
    ==========  ==================  ========================

    Args:
        storage_path (str): Path to the new dsch storage (backend-specific).
        schema_node: Top-level schema node for the dsch storage. See
            :mod:`dsch.schema` for details.
        backend (str): Backend to use for the new dsch storage.

    Returns:
        Storage object.
    """
    if not backend:
        backend = _autodetect_backend(storage_path)
    backend_module = importlib.import_module('dsch.backends.' + backend)
    return backend_module.Storage(storage_path=storage_path,
                                  schema_node=schema_node)


def create_from(storage_path, source_storage, backend=None):
    """Create a new dsch storage by copying from an existing one.

    The new storage is created, just like with :func:`create`, but the schema
    is automatically copied from the given ``source_storage``. In addition, all
    data currently stored in ``source_storage`` is also copied.

    If no backend is specified, it is detected automatically by interpreting
    the ``storage_path``, e.g. via a file extension. For details, see
    :func:`create`.

    Args:
        storage_path (str): Path to the new dsch storage (backend-specific).
        source_storage: dsch storage to copy schema and data from.
        backend (str): Backend to use for the new dsch storage.

    Returns:
        Newly created dsch storage.
    """
    storage = create(storage_path, source_storage.schema_node, backend)
    storage.data.load_from(source_storage.data)
    return storage


def load(storage_path, backend=None, required_schema=None,
         required_schema_hash=None, force=False):
    """Load a dsch storage from the given path.

    Normally, the correct backend is detected automatically by interpreting the
    ``storage_path``, e.g. via a file extension. Alternatively, the backend can
    be forced to a desired value by additionally passing a ``backend``
    argument.

    The ``required_schema`` or ``required_schema_hash`` arguments can be used to
    ensure that the loaded storage uses a specific schema. The former is used
    to supply a schema object while the latter must be the SHA256 hash of the
    required schema JSON, as can be determined by
    :meth:`.storage.Storage.schema_hash`. If the loaded storage uses a
    different schema, an exception is raised.

    In addition, the loaded storage is validated automatically, unless
    ``force`` is set to ``True``.
    This ensures that the loaded data really conforms to the desired schema, so
    that following code, e.g. for data evaluation, can safely depend on the
    structure, datatypes and met constraints.

    Args:
        storage_path (str): Path to the dsch storage (backend-specific).
        backend (str): Backend to be used. By default, perform auto-detection.
        required_schema (dsch.schema.SchemaNode): Top-level schema node of the
            required schema.
        required_schema_hash (str): SHA256 hash of the required schema.
        force (bool): If ``True``, the automatic validation step is skipped.

    Returns:
        Storage object.

    Raises:
        dsch.exceptions.InvalidSchemaError: if the loaded storage's schema does
            not match the schema specified through ``required_schema`` or
            ``required_schema_hash``.
        dsch.exceptions.ValidationError: if ``force`` was not ``True`` and
            validation failed with a regular node as the top-level node.
        dsch.exceptions.SubnodeValidationError: if ``force`` was not ``True``
            and validation failed with a :class:`~dsch.data.Compilation` or
            :class:`dsch.data.List` as the top-level schema node.
    """
    if not backend:
        backend = _autodetect_backend(storage_path)
    backend_module = importlib.import_module('dsch.backends.' + backend)
    storage = backend_module.Storage(storage_path=storage_path)
    if required_schema and storage.schema_hash() != required_schema.hash():
        raise exceptions.InvalidSchemaError(required_schema.hash(),
                                            storage.schema_hash())
    if required_schema_hash and storage.schema_hash() != required_schema_hash:
        raise exceptions.InvalidSchemaError(required_schema_hash,
                                            storage.schema_hash())
    if not force:
        storage.validate()
    return storage


class PseudoStorage:
    """Provide abstraction between storages and data nodes.

    ``PseudoStorage`` provides an easy way to manage data with dsch when it is
    either in a :class:`~dsch.storage.Storage` or in a :mod:`~dsch.data`-node.
    Being able to handle both variants with the same code base is relevant
    especially for libraries that support schema extension.

    Schema extension means that when a library defines its dsch schema, other
    code (higher-level libraries or applications) may incorporate it into a
    broader schema. That way, the application can use a single schema for its
    specific purposes while retaining compatibility with the library's schema.
    To support this, the library must be able to handle its own schema being at
    either the top level or a subordinate level inside the storage. In the
    former case, tasks like the creation of the storage are part of the
    library's responsibility, while in the latter, they are not.

    ``PseudoStorage`` can be initialized with either a :class:`str` or a
    data node (from :mod:`dsch.data`) object. If a string is given, it is
    interpreted as a ``storage_path``, just like :func:`create` and
    :func:`load`. The corresponding storage is made available through
    :attr:`storage` and its data through :attr:`data`. Alternatively, if a data
    node is given during initialization, it is direcly made available through
    :attr:`data`. In that case :attr:`storage` is set to ``None``, indicating
    that the ``PseudoStorage`` is not managing the actual storage object.

    In addition, the object can be used as a context manager using the ``with``
    statement. This causes the data to be made available (e.g. by opening a
    file) when the context is entered and to be cleared (i.e. :attr:`data` set
    to ``None``) when the context is left. When an entire storage is managed
    (i.e. the object was initialized with a string), the storage is also saved
    (see :meth:`dsch.storage.FileStorage.save`) if applicable.
    If usage as a context manager is not desired, the same functionality is
    also exposed as :meth:`open` and :meth:`close`.

    To support older versions of a schema once schema changes cannot be
    avoided, ``schema_alternatives`` can be passed on object creation. If an
    existing storage is loaded (or a data node is passed) based on an
    alternative schema, no exception is raised and the process continues
    normally. Note that new storages are always created with the given
    ``schema_node``.
    The actual top-level schema node of the available data is made
    available through :attr:`schema_node`, so it can be easily inspected by
    subsequent code. ``schema_alternatives`` is an iterable containing either
    the :class:`~dsch.schema.SchemaNode` object or the corresponding hash for
    every supported schema.

    Attributes:
        data: Data node, corresponding to the top-level node of the schema.
        storage: A dsch storage object, if the ``PseudoStorage`` was
            initialized with a string.
        schema_node: A :class:`~dsch.schema.SchemaNode` representing the
            top-level schema node of the actual data loaded. This is either the
            same as the ``schema_node`` specified on object creation, or one of
            the schemas listed in ``schema_alternatives`` upton object
            creation.
    """

    def __init__(self, data_storage, schema_node, defer_open=False,
                 schema_alternatives=None):
        """Create a pseudo-storage for data access abstraction.

        Args:
            storage: An :class:`str` or an existing data node corresponding to
                the ``schema_node``. If a string is given, the abstractor
                creates or loads a storage, using the string as its
                ``storage_path``. If a data node is given, it is considered a
                sub-node of a broader storage.
            schema_node: The top-level node of the desired schema. This is used
                when creating a new storage, and for validating loaded storages
                and data nodes.
            defer_open: If ``True``, the desired data is not made available
                (potentially by creating or loading a storage) until the
                runtime context is entered or :meth:`open` is called
                explicitly.
            schema_alternatives: An iterable of schemas that are accepted
                besides what is given as ``schema_node``. Allows support of
                previous versions of the schema. Every entry may be either a
                :class:`~dsch.schema.SchemaNode` or a :class:`str` value
                representing a :meth:`~dsch.schema.SchemaNode.hash()`.
        """
        self._data_storage = data_storage
        self._schema_node = schema_node
        self._schema_alternatives = []
        if schema_alternatives:
            for alt in schema_alternatives:
                if isinstance(alt, schema.SchemaNode):
                    self._schema_alternatives.append(alt.hash())
                else:
                    self._schema_alternatives.append(alt)

        self.data = None
        self.storage = None
        self.schema_node = None
        if not defer_open:
            self.open()

    def close(self):
        """Finalize data access.

        This saves and closes the corresponding storage, if applicable.
        Afterwards, the data is no longer available through :attr:`data`.
        """
        if self.storage:
            if hasattr(self.storage, 'save') and callable(self.storage.save):
                self.storage.save()
            self.storage = None
        self.data = None
        self.schema_node = None

    def open(self):
        """Make the desired data available as :attr:`data`.

        If necessary, this loads or creates a new storage.
        """
        if self.data is not None:
            raise RuntimeError('PseudoStorage is already open.')
        if isinstance(self._data_storage, str):
            if os.path.exists(self._data_storage):
                self.storage = load(self._data_storage)
                self._verify_schema(self.storage.schema_node)
            else:
                self.storage = create(self._data_storage,
                                      schema_node=self._schema_node)
            self.data = self.storage.data
        else:
            self.storage = None
            self._verify_schema(self._data_storage.schema_node)
            self.data = self._data_storage
        self.schema_node = self.data.schema_node

    def __enter__(self):
        if self.data is None:
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _verify_schema(self, schema_node):
        schema_hash = schema_node.hash()
        if (schema_hash != self._schema_node.hash() and
                schema_hash not in self._schema_alternatives):
            raise exceptions.InvalidSchemaError(self._schema_node.hash(),
                                                schema_hash)


def _autodetect_backend(storage_path):
    """Find the backend name corresponding to the given storage path.

    Args:
        storage_path (str): Path to the dsch storage.

    Returns:
        str: Corresponding backend name.

    Raises:
        dsch.exceptions.AutodetectBackendError: if automatic detection fails.
    """
    if storage_path == '::inmem::':
        return 'inmem'
    elif storage_path.endswith('.npz'):
        return 'npz'
    elif storage_path.endswith(('.h5', '.hdf5')):
        return 'hdf5'
    if storage_path.endswith('.mat'):
        return 'mat'
    else:
        raise exceptions.AutodetectBackendError(storage_path)
