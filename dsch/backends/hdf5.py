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
        try:
            del self._parent[self._dataset_name]
        except KeyError:
            # If the dataset has not been created yet, that's also okay.
            pass
        self._storage = self._parent.create_dataset(self._dataset_name,
                                                    data=new_value)


class Array(_ItemNode):
    """Array-type data node for the HDF5 backend."""

    def __getitem__(self, key):
        """Pass slicing/indexing operations directly to HDF5 dataset."""
        return self._storage[key]

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        try:
            del self._parent[self._dataset_name]
        except KeyError:
            # If the dataset has not been created yet, that's also okay.
            pass

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

    def resize(self, size):
        """Resize the array to the desired size.

        Args:
            size (tuple): Desired array size.
        """
        self._storage.resize(size)

    def __setitem__(self, key, value):
        """Pass slicing/indexing operations directly to HDF5 dataset."""
        self._storage[key] = value

    def validate(self):
        """Validate the node value against the schema node specification.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            :exc:`dsch.schema.ValidationError`: if validation fails.
        """
        independent_values = []
        node_names = self.schema_node.depends_on or []
        for node_name in node_names:
            if node_name:
                independent_values.append(getattr(self.parent, node_name)
                                          .value)
        self.schema_node.validate(self.value, independent_values)

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        if self._storage is None:
            raise RuntimeError('Empty data node has no value.')
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

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        if self._storage is None:
            raise RuntimeError('Empty data node has no value.')
        return bool(self._storage.value)


class Compilation(data.Compilation):
    """Compilation-type data node for the HDF5 backend."""

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


class Date(_ItemNode):
    """Date-type data node for the HDF5 backend."""

    def _init_new(self, new_params):
        """Initialize new Date data node.

        For :class:`Date`, the corresponding HDF5 dataset is only created if
        the corresponding schema node's :attr:`dsch.schema.Date.set_on_create`
        is ``True``. Otherwise, the dataset is left empty and can be filled by
        calling meth:`replace`.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.replace(datetime.date.today())

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

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return datetime.date(*self._storage.value.tolist())


class DateTime(_ItemNode):
    """DateTime-type data node for the HDF5 backend."""

    def _init_new(self, new_params):
        """Initialize new DateTime data node.

        For :class:`DateTime`, the corresponding HDF5 dataset is only created
        if the corresponding schema node's
        :attr:`dsch.schema.DateTime.set_on_create` is ``True``. Otherwise, the
        dataset is left empty and can be filled by calling meth:`replace`.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.replace(datetime.datetime.now())

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

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return datetime.datetime(*self._storage.value.tolist())


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
        try:
            del self._parent[self._dataset_name]
        except KeyError:
            # If the dataset has not been created yet, that's also okay.
            pass

        self._storage = self._parent.create_dataset(
            self._dataset_name,
            data=new_value,
            dtype=self.schema_node.dtype,
        )

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        if self._storage is None:
            raise RuntimeError('Empty data node has no value.')
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
        self._schema_from_json(self._storage.attrs['dsch_schema'])
        if isinstance(self.schema_node, schema.Compilation):
            data_storage = self._storage
        else:
            # If the top-level node is not a Compilation, apply the default
            # name 'dsch_data'.
            data_storage = self._storage['dsch_data']
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None,
                                               data_storage=data_storage)

    def _new(self):
        """Create a new file at :attr:`storage_path`."""
        self._storage = h5py.File(self.storage_path, 'x')
        self._storage.attrs['dsch_schema'] = self._schema_to_json()
        if isinstance(self.schema_node, schema.Compilation):
            new_params = {'name': '', 'parent': self._storage}
        else:
            # If the top-level node is not a Compilation, apply the default
            # name 'dsch_data'.
            new_params = {'name': 'dsch_data', 'parent': self._storage}
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__, None,
                                               new_params=new_params)

    def save(self):
        """Save the current data to the file in :attr:`storage_path`.

        Note: This does not perform any validation, so the created file is
        *not* guaranteed to fulfill the schema's constraints.
        """
        self._storage.flush()


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

    def _init_new(self, new_params):
        """Initialize new, empty List data node.

        This creates a new HDF5 group to hold the List's sub-nodes.

        Args:
            new_params (dict): Dict including the HDF5 group name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        self._storage = new_params['parent'].create_group(new_params['name'])


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

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        if self._storage is None:
            raise RuntimeError('Empty data node has no value.')
        return self._storage.value


class Time(_ItemNode):
    """Time-type data node for the HDF5 backend."""

    def _init_new(self, new_params):
        """Initialize new Time data node.

        For :class:`Time`, the corresponding HDF5 dataset is only created if
        the corresponding schema node's :attr:`dsch.schema.Time.set_on_create`
        is ``True``. Otherwise, the dataset is left empty and can be filled by
        calling meth:`replace`.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.replace(datetime.datetime.now().time())

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

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return datetime.time(*self._storage.value.tolist())
