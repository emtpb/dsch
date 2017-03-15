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
from . import helpers


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
    npz         NumPy .npz file     Path to regular file
    ==========  ==================  ========================

    Args:
        storage_path (str): Path to the new dsch storage.
        schema_node: Top-level schema node for the dsch storage. See
            :mod:`dsch.schema` for details.
        backend (str): Backend to use for the new dsch storage.

    Returns:
        Storage object.
    """
    if not backend:
        backend = _autodetect_backend(storage_path)
    return helpers.backend_module(backend).Storage(storage_path=storage_path,
                                                   schema_node=schema_node)


def load(storage_path, backend=None):
    """Load a dsch storage from the given path.

    Normally, the correct backend is detected automatically by interpreting the
    ``storage_path``, e.g. via a file extension. Alternatively, the backend can
    be forced to a desired value by additionally passing a ``backend``
    argument.

    Args:
        storage_path (str): Path to the dsch storage to load.
        backend (str): Backend to be used. By default, perform auto-detection.

    Returns:
        Storage object.
    """
    if not backend:
        backend = _autodetect_backend(storage_path)
    return helpers.backend_module(backend).Storage(storage_path=storage_path)


def _autodetect_backend(storage_path):
    """Find the backend name corresponding to the given storage path.

    Args:
        storage_path (str): Path to the dsch storage.

    Returns:
        str: Corresponding backend name.

    Raises:
        ValueError: if automatic detection fails.
    """
    if storage_path.endswith('.npz'):
        return 'npz'
    elif storage_path.endswith(('.h5', '.hdf5')):
        return 'hdf5'
    else:
        raise ValueError('Could not automatically detect backend for '
                         '"{}".'.format(storage_path))
