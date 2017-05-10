import numpy as np
import scipy.io as sio
from .. import data
from . import npz


class Array(npz.Array):
    """Array-type data node for the mat backend."""
    pass


class Bool(npz.Bool):
    """Bool-type data node for the mat backend."""
    pass


class Compilation(npz.Compilation):
    """Compilation-type data node for the mat backend."""

    def _init_from_storage(self, data_storage):
        """Initialize Compilation from the given data storage object.

        This recursively initializes data nodes for all sub-nodes, using the
        given data storage object.

        The mat backend's implementation uses NumPy structured arrays, as per
        https://docs.scipy.org/doc/scipy/reference/tutorial/io.html, causing
        the Compilation to be represented as a struct in MATLAB.

        Args:
            data_storage (:class:`numpy.ndarray`): Backend-specific data
                storage object to load.
        """
        # For Compilation inside Compilation, scipy.io.loadmat seems to wrap
        # the inner Compilation in an additional scalar ndarray with dtype
        # object. This must be ignored, if present.
        if data_storage.shape == ():
            data_storage = data_storage[()]

        for node_name, subnode in self.schema_node.subnodes.items():
            self._subnodes[node_name] = data.data_node_from_schema(
                subnode, self.__module__, self,
                data_storage=data_storage[node_name])


class Date(npz.Date):
    """Date-type data node for the mat backend."""
    pass


class DateTime(npz.DateTime):
    """DateTime-type data node for the mat backend."""
    pass


class List(npz.List):
    """List-type data node for the mat backend."""

    def _init_from_storage(self, data_storage):
        """Initialize List from the given data storage object.

        This recursively initializes data nodes for all sub-nodes, using the
        given data storage object.

        The mat backend's implementatino uses NumPy object arrays, as per
        https://docs.scipy.org/doc/scipy/reference/tutorial/io.html, causing
        the List to be represented as a cell array in MATLAB.

        Args:
            data_storage (:class:`numpy.ndarray`): Backend-specific data
                storage object to load.
        """
        for field in data_storage:
            node_storage = field
            subnode = data.data_node_from_schema(self.schema_node.subnode,
                                                 self.__module__, self,
                                                 data_storage=node_storage)
            self._subnodes.append(subnode)

    def save(self):
        """Export the node data as a data storage object.

        For the mat backend, List data is represented as a NumPy object array,
        i.e. a :class:`numpy.ndarray` with ``dtype=numpy.object``.

        Returns:
            dict: Data storage object with the node's data.
        """
        data_storage = np.zeros((len(self._subnodes),), dtype=np.object)
        for idx, node in enumerate(self._subnodes):
            data_storage[idx] = node.save()
        return data_storage


class Scalar(npz.Scalar):
    """Scalar-type data node for the mat backend."""
    pass


class Storage(npz.Storage):
    """Interface to ``.mat`` files.

    Provides access to an ``.mat`` file via dsch, i.e. reading from and writing
    to such a file.

    Attributes:
        storage_path (str): Path to the current storage.
        schema_node: Top-level schema node used for the stored data.
        data: Top-level data node, providing access to all managed data.
    """

    def _load(self):
        """Load an existing file from :attr:`storage_path`."""
        file_ = sio.loadmat(self.storage_path, squeeze_me=True)
        self._schema_from_json(file_['schema'])
        data_storage = file_.get('data', None)
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None,
                                               data_storage=data_storage)

    def save(self):
        """Save the current data to the file in :attr:`storage_path`.

        Note: This does not perform any validation, so the created file is
        *not* guaranteed to fulfill the schema's constraints.
        """
        store_data = {'schema': self._schema_to_json()}
        output_data = self.data.save()
        if output_data is not None:
            store_data['data'] = output_data
        sio.savemat(self.storage_path, store_data)


class String(npz.String):
    """String-type data node for the mat backend."""
    pass


class Time(npz.Time):
    """Time-type data node for the mat backend."""
    pass
