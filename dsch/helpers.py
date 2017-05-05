"""Helper module for functions not directly tied to the project structure."""
import importlib


def backend_module(backend_name):
    """Return the backend module corresponding to the given name.

    Note: The module is automatically imported if it is accessed for the first
    time.

    Args:
        backend_name (str): Name of the backend module to find.

    Returns:
        module: Backend module for the given name.
    """
    return importlib.import_module('dsch.backends.' + backend_name)
