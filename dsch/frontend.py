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


def load(storage_path, backend=None, require_schema=None, force=False):
    """Load a dsch storage from the given path.

    Normally, the correct backend is detected automatically by interpreting the
    ``storage_path``, e.g. via a file extension. Alternatively, the backend can
    be forced to a desired value by additionally passing a ``backend``
    argument.

    The ``require_schema`` argument can be used to ensure that the loaded
    storage uses a specific schema. The value must be the SHA256 hash of the
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
        require_schema (str): SHA256 hash of the required schema.
        force (bool): If ``True``, the automatic validation step is skipped.

    Returns:
        Storage object.

    Raises:
        RuntimeError: if the SHA256 hash given to ``require_schema`` did not
            match.
        :class:`.schema.ValidationError` or
            :class:`.data.SubnodeValidationError`: if validation failed and
            ``force`` was not ``True``.
    """
    if not backend:
        backend = _autodetect_backend(storage_path)
    backend_module = importlib.import_module('dsch.backends.' + backend)
    storage = backend_module.Storage(storage_path=storage_path)
    if require_schema and storage.schema_hash() != require_schema:
        raise RuntimeError('Loaded schema does not match the required schema.')
    if not force:
        storage.validate()
    return storage


def _autodetect_backend(storage_path):
    """Find the backend name corresponding to the given storage path.

    Args:
        storage_path (str): Path to the dsch storage.

    Returns:
        str: Corresponding backend name.

    Raises:
        ValueError: if automatic detection fails.
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
        raise ValueError('Could not automatically detect backend for '
                         '"{}".'.format(storage_path))
