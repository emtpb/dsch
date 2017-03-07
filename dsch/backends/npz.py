"""dsch backend for NumPy's npz format.

This backend provides support for NumPy's npz format. For details, see
:func:`numpy.savez`, :func:`numpy.load` and the `corresponding NumPy
enhancement proposal <https://docs.scipy.org/doc/numpy/neps/npy-format.html>`_.
"""
import numpy as np
from .. import data


class Bool(data.ItemNode):
    """Bool-type data node for the npz backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self.storage = np.array([new_value], dtype='bool')

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return bool(self.storage)
