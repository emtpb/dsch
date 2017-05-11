"""dsch backend for HDF5 files via h5py.

This backend provides support for the HDF5 file format. :mod:`h5py` is used as
the interface.
"""
import datetime
import h5py
import numpy as np
from .. import data, schema, storage


class _ItemNode(data.ItemNode):
    """Common base class for data nodes for the HDF5 backend."""

    def clear(self):
        """Clear the data that is held by this data node.

        This removes the corresponding storage object entirely, causing the
        data node to be :attr:`empty` afterwards.
        """
        super().clear()
        try:
            del self._parent[self._dataset_name]
        except KeyError:
            # If the dataset has not been created yet, that's also okay.
            pass

    def _init_from_storage(self, data_storage):
        """Create a new data node from a data storage object.

        This initializes the data node using the given data storage object.

        Args:
            data_storage (:class:`h5py.Dataset`): Data storage object to be
                imported.
        """
        super()._init_from_storage(data_storage)
        self._dataset_name = data_storage.name.split('/')[-1]
        self._parent = data_storage.parent

    def _init_new(self, new_params):
        """Initialize new, empty data node.

        The new data node is generally initialized without data, i.e. no
        storage data object exists. Use :meth:`replace` to apply the desired
        value.

        The HDF5 dataset name and parent are given as ``new_params['parent']``
        and ``new_params['name']``.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        self._dataset_name = new_params['name']
        self._parent = new_params['parent']

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self.clear()
        self._storage = self._parent.create_dataset(self._dataset_name,
                                                    data=new_value)


class Array(data.Array, _ItemNode):
    """Array-type data node for the HDF5 backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self.clear()

        if self.schema_node.max_shape:
            if self.schema_node.max_shape == self.schema_node.min_shape:
                # For equal shape constraints, disable resizing entirely.
                maxshape = None
            else:
                maxshape = self.schema_node.max_shape
        else:
            # Make all dimensions arbitrarily resizable by default.
            maxshape = (None,) * self.schema_node.ndim

        self._storage = self._parent.create_dataset(
            self._dataset_name,
            data=new_value,
            dtype=self.schema_node.dtype,
            maxshape=maxshape,
        )

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return self._storage.value


class Bool(_ItemNode):
    """Bool-type data node for the HDF5 backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        if self._storage:
            self._storage[()] = new_value
        else:
            self._storage = self._parent.create_dataset(self._dataset_name,
                                                        data=new_value)

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return bool(self._storage.value)


class Compilation(data.Compilation):
    """Compilation-type data node for the HDF5 backend."""

    def _init_from_storage(self, data_storage):
        """Initialize Compilation from the given data storage object.

        This recursively initializes data nodes for all sub-nodes, using the
        given data storage object.

        Args:
            data_storage (dict): Backend-specific data storage object to load.
        """
        for node_name, subnode in self.schema_node.subnodes.items():
            if node_name in data_storage:
                self._subnodes[node_name] = data.data_node_from_schema(
                    subnode, self.__module__, self,
                    data_storage=data_storage[node_name])
            else:
                new_params_sub = {'name': node_name, 'parent': data_storage}
                self._subnodes[node_name] = data.data_node_from_schema(
                    subnode, self.__module__, self, new_params=new_params_sub)

    def _init_new(self, new_params):
        """Initialize new, empty Compilation.

        This recursively initializes new data nodes for all sub-nodes, but does
        not import any existing data.
        All sub-nodes are created in a new HDF5 group that represents the
        Compilation. This can be disabled by passing
        ``new_params['name'] == ''``, resulting in the given HDF5 group to be
        directly used as the parent, e.g. when the Compilation corresponds to
        the HDF5 root group.

        Args:
            new_params (dict): Dict including the HDF5 group name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        if new_params['name']:
            comp_group = new_params['parent'].create_group(new_params['name'])
        else:
            comp_group = new_params['parent']

        for node_name, subnode in self.schema_node.subnodes.items():
            new_params_sub = {'name': node_name, 'parent': comp_group}
            self._subnodes[node_name] = data.data_node_from_schema(
                subnode, self.__module__, self, new_params=new_params_sub)


class Date(data.Date, _ItemNode):
    """Date-type data node for the HDF5 backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        repr_data = np.array([new_value.year, new_value.month, new_value.day],
                             dtype='int')
        super().replace(repr_data)

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return datetime.date(*self._storage.value.tolist())


class DateTime(data.DateTime, _ItemNode):
    """DateTime-type data node for the HDF5 backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        repr_data = np.array([new_value.year, new_value.month, new_value.day,
                              new_value.hour, new_value.minute,
                              new_value.second, new_value.microsecond],
                             dtype='int')
        super().replace(repr_data)

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return datetime.datetime(*self._storage.value.tolist())


class List(data.List):
    """List-type data node for the HDF5 backend."""

    def append(self, value=None):
        """Append a new data node to the list.

        If a ``value`` is given, it is automatically applied to the new data
        node. Otherwise, an empty data node is created, which can be useful
        especially for Lists of Compilations.

        Note: This works with actual data values!

        Args:
            value: Value to be added to the list.
        """
        new_params = {'name': 'item_{}'.format(len(self)),
                      'parent': self._storage}
        subnode = data.data_node_from_schema(self.schema_node.subnode,
                                             self.__module__, self,
                                             new_params=new_params)
        self._subnodes.append(subnode)
        if value is not None:
            subnode.replace(value)

    def clear(self):
        """Clear all subnodes."""
        for name in self._storage.keys():
            del self._storage[name]
        super().clear()

    def _init_from_storage(self, data_storage):
        """Create a new data node from a data storage object.

        This initializes the data node using the given data storage object.

        Args:
            data_storage (:class:`h5py.Dataset`): Data storage object to be
                imported.
        """
        super()._init_from_storage(data_storage)
        self._storage = data_storage

    def _init_new(self, new_params):
        """Initialize new, empty List data node.

        This creates a new HDF5 group to hold the List's sub-nodes.

        Args:
            new_params (dict): Dict including the HDF5 group name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        self._storage = new_params['parent'].create_group(new_params['name'])


class Scalar(_ItemNode):
    """Scalar-type data node for the HDF5 backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self.clear()
        self._storage = self._parent.create_dataset(
            self._dataset_name,
            data=new_value,
            dtype=self.schema_node.dtype,
        )

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return self._storage.value


class Storage(storage.FileStorage):
    """Interface to HDF5 files.

    Provides access to an HDF5 file via dsch, i.e. reading from and writing
    to such a file.

    Attributes:
        storage_path: Path to the current storage file. This gets set
            automatically when loading from or saving to a new target file.
        schema_node: The top-level schema node for the file.
        data: The top-level data node, corresponding to :attr:`schema_node`.
    """

    def _load(self):
        """Load an existing file from :attr:`storage_path`."""
        self._storage = h5py.File(self.storage_path)
        self.schema_node = schema.node_from_json(
            self._storage.attrs['dsch_schema'])
        if isinstance(self.schema_node, schema.Compilation):
            data_storage = self._storage
        else:
            # If the top-level node is not a Compilation, apply the default
            # name 'dsch_data'.
            if 'dsch_data' in self._storage:
                data_storage = self._storage['dsch_data']
            else:
                new_params = {'name': 'dsch_data', 'parent': self._storage}
                self.data = data.data_node_from_schema(self.schema_node,
                                                       self.__module__, None,
                                                       new_params=new_params)
                return
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None,
                                               data_storage=data_storage)

    def _new(self):
        """Create a new file at :attr:`storage_path`."""
        self._storage = h5py.File(self.storage_path, 'x')
        self._storage.attrs['dsch_schema'] = self.schema_node.to_json()
        if isinstance(self.schema_node, schema.Compilation):
            new_params = {'name': '', 'parent': self._storage}
        else:
            # If the top-level node is not a Compilation, apply the default
            # name 'dsch_data'.
            new_params = {'name': 'dsch_data', 'parent': self._storage}
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None,
                                               new_params=new_params)

    def _save(self):
        """Save the current data to the file in :attr:`storage_path`."""
        self._storage.flush()


class String(_ItemNode):
    """String-type data node for the HDF5 backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        if self._storage:
            self._storage[()] = new_value
        else:
            self._storage = self._parent.create_dataset(self._dataset_name,
                                                        data=new_value)

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return self._storage.value


class Time(data.Time, _ItemNode):
    """Time-type data node for the HDF5 backend."""

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        repr_data = np.array([new_value.hour, new_value.minute,
                              new_value.second, new_value.microsecond],
                             dtype='int')
        super().replace(repr_data)

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        return datetime.time(*self._storage.value.tolist())
