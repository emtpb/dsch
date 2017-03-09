"""dsch backends package.

Dsch supports multiple storage backends through a plugin architecture. Each
backend must be implemented in a separate module inside the ``backends``
package, deriving from the classes in :mod:`dsch.data` and implementing the
specific storage functionality.

After import, :data:`available_backends` is a tuple of all available backend's
names.
"""
import os
import pkgutil


# Automatically discover all available backends.
_pkg_path = os.path.dirname(__file__)
available_backends = tuple([mod[1] for mod in
                            pkgutil.iter_modules([_pkg_path])])
