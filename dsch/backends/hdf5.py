"""dsch backend for HDF5 files via h5py.

This backend provides support for the HDF5 file format. :mod:`h5py` is used as
the interface.
"""
import h5py
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

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        del self._parent[self._dataset_name]
        self.storage = self._parent.create_dataset(self._dataset_name,
                                                   data=new_value)


class Bool(_ItemNode):
    """Bool-type data node for the HDF5 backend."""

    def _init_new(self, new_params):
        """Initialize new, empty Bool data node.

        Creates a new HDF5 dataset as the data storage for this node. The HDF5
        dataset name and parent are given as ``new_params['parent']`` and
        ``new_params['name']``.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        self._dataset_name = new_params['name']
        self._parent = new_params['parent']
        self.storage = self._parent.create_dataset(self._dataset_name,
                                                   dtype='bool', shape=())

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return bool(self.storage.value)


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
                subnode, self.__module__, new_params=new_params_sub)


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
        self.storage = h5py.File(self.storage_path)
        self._schema_from_json(self.storage.attrs['dsch_schema'])
        if isinstance(self.schema_node, schema.Compilation):
            data_storage = self.storage
        else:
            # If the top-level node is not a Compilation, apply the default
            # name 'dsch_data'.
            data_storage = self.storage['dsch_data']
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__,
                                               data_storage=data_storage)

    def _new(self):
        """Create a new file at :attr:`storage_path`."""
        self.storage = h5py.File(self.storage_path, 'x')
        self.storage.attrs['dsch_schema'] = self._schema_to_json()
        if isinstance(self.schema_node, schema.Compilation):
            new_params = {'name': '', 'parent': self.storage}
        else:
            # If the top-level node is not a Compilation, apply the default
            # name 'dsch_data'.
            new_params = {'name': 'dsch_data', 'parent': self.storage}
        self.data = data.data_node_from_schema(self.schema_node,
                                               self.__module__,
                                               new_params=new_params)

    def save(self):
        """Save the current data to the file in :attr:`storage_path`.

        Note: This does not perform any validation, so the created file is
        *not* guaranteed to fulfill the schema's constraints.
        """
        self.storage.flush()


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
                                             self.__module__,
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

    def _init_new(self, new_params):
        """Initialize new, empty String data node.

        Creates a new HDF5 dataset as the data storage for this node. The HDF5
        dataset name and parent are given as ``new_params['parent']`` and
        ``new_params['name']``.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        self._dataset_name = new_params['name']
        self._parent = new_params['parent']
        self.storage = self._parent.create_dataset(self._dataset_name,
                                                   data='')

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        schema node, not on the selected backend.

        Returns:
            Node data.
        """
        return self.storage.value
