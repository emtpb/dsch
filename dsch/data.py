"""dsch data representation.

In dsch, data is structured according to a given schema. The data is then
represented as a hierarchical structure of data nodes, each of which
corresponds to a node in the schema. This allows subsequent validation against
the schema.

The data nodes are also responsible for storing the data. Since dsch is built
to support multiple storage backends, there are specific data node classes
implementing the respective functionality. The classes in this module provide
common functionality and are intended to be used as base classes.
Different backends are implemented in the ``backends`` package.
"""
import importlib


class Compilation:
    """Compilation data node.

    :class:`Compilation` is the base class for compilation-type data nodes,
    providing common functionality and the common interface. Subclasses may add
    functionality depending on the backend.

    Attributes:
        schema_node: The schema node that this data node is based on.
        parent: Parent data node object (``None`` if this is the top-level data
            node).
        empty: Data absence flag. ``True`` if no data is present.
    """

    def __init__(self, schema_node, parent, data_storage=None,
                 new_params=None):
        """Initialize compilation node from a given schema node.

        When ``data_storage`` is given, the Compilation and all its sub-nodes
        are created from that data storage object, i.e. loading existing data.
        Otherwise, a new Compilation is created with empty sub-nodes, possibly
        using optional ``new_params`` for creation.
        Note that the data node instances for the sub-nodes are created in both
        cases.

        Note: Both ``data_storage`` and ``new_params`` are backend-specific.

        Args:
            schema_node: Schema node to create the data node for.
            parent: Parent data node object (``None`` if this is the top-level
                data node).
            data_storage: Backend-specific data storage object to load.
            new_params: Backend-specific metadata for data node creation.
        """
        self.schema_node = schema_node
        self.parent = parent
        self._subnodes = {}
        if data_storage is not None:
            self._init_from_storage(data_storage)
        else:
            self._init_new(new_params)

    def __dir__(self):
        """Include sub-nodes in :meth:`dir`."""
        attrs = super().__dir__()
        attrs.extend(self._subnodes.keys())
        return attrs

    @property
    def empty(self):
        """Check whether the Compilation is currently empty.

        A Compilation is considered empty when all individual sub-nodes are
        empty.

        Returns:
            bool: ``True`` if the data node is empty, ``False`` otherwise.
        """
        for node in self._subnodes.values():
            if not node.empty:
                return False
        return True

    def __getattr__(self, attr_name):
        """Return sub-nodes via the dot-attribute syntax.

        This returns the entire sub-node object, not just the node value.
        """
        return self._subnodes[attr_name]

    def _init_from_storage(self, data_storage):
        """Initialize Compilation from the given data storage object.

        This recursively initializes data nodes for all sub-nodes, using the
        given data storage object.

        The default implementation expects a dict, where keys are the
        Compilation's sub-node names and values are the corresponding data
        storage objects. Backend-specific subclasses may change this, if
        required.

        Args:
            data_storage (dict): Backend-specific data storage object to load.
        """
        for node_name, subnode in self.schema_node.subnodes.items():
            self._subnodes[node_name] = data_node_from_schema(
                subnode, self.__module__, self,
                data_storage=data_storage[node_name])

    def _init_new(self, new_params):
        """Initialize new, empty Compilation.

        This recursively initializes new data nodes for all sub-nodes, but does
        not import any existing data.

        Args:
            new_params: Backend-specific metadata for data node creation.
        """
        for node_name, subnode in self.schema_node.subnodes.items():
            self._subnodes[node_name] = data_node_from_schema(
                subnode, self.__module__, self)

    def replace(self, new_value):
        """Replace the current compilation values with new ones.

        The new values must be specified as a :class:`dict`, where the key
        corresponds to the compilation field name.

        For :class:`Compilation`, this method is effectively a shorthand for
        calling :meth:`ItemNode.replace` on all fields specified in the given
        dict.

        Args:
            new_value (dict): Mapping of field names to new values.
        """
        for key, value in new_value.items():
            self._subnodes[key].replace(value)

    def validate(self):
        """Recursively validate all sub-node values.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            :exc:`dsch.schema.ValidationError`: if validation fails.
        """
        for node in self._subnodes.values():
            node.validate()


class ItemNode:
    """Generic data node.

    :class:`ItemNode` is the base class for data nodes, providing common
    functionality and the common interface. Subclasses may add functionality
    depending on the node type and backend (e.g. compression settings).

    Note that this is only the base class for item nodes, i.e. nodes that
    directly hold data. Collection nodes, i.e. :class:`Compilation` and
    :class:`List` are *not* based on this class.

    Attributes:
        schema_node: The schema node that this data node is based on.
        parent: Parent data node object (``None`` if this is the top-level data
            node).
        empty: Data absence flag. ``True`` if no data is present.
        value: Actual node data, independent of the backend in use.
    """

    def __init__(self, schema_node, parent, data_storage=None,
                 new_params=None):
        """Initialize data node from a given schema node.

        Note: Both ``data_storage`` and ``new_params`` are backend-specific.

        Args:
            schema_node: Schema node to create the data node for.
            parent: Parent data node object (``None`` if this is the top-level
                data node).
            data_storage: Backend-specific data storage object to load.
            new_params: Backend-specific metadata for data node creation.
        """
        self.schema_node = schema_node
        self.parent = parent
        self._storage = None
        if data_storage is not None:
            self._init_from_storage(data_storage)
        else:
            self._init_new(new_params)

    @property
    def empty(self):
        """Check whether the data node is currently empty.

        A data node is considered empty when no corresponding storage object
        exists. For applying a new value, see :meth:`replace`.

        Returns:
            bool: ``True`` if the data node is empty, ``False`` otherwise.
        """
        return self._storage is None

    def _init_from_storage(self, data_storage):
        """Create a new data node from a data storage object.

        This initializes the data node using the given data storage object.

        The default implementation simply assigns ``data_storage`` as the data
        item's data storage object, which should work for most backends.
        Specific subclasses may override this behaviour, if required.

        Args:
            data_storage: Backend-specific data storage object to load.
        """
        self._storage = data_storage

    def _init_new(self, new_params):
        """Initialize new, empty data node.

        The default implementation does nothing. It is provided for interface
        consistency and for possible overriding by backend-specific subclasses.

        Args:
            new_params: Backend-specific metadata for data node creation.
        """
        pass

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        raise NotImplementedError('To be implemented in subclass.')

    def validate(self):
        """Validate the node value against the schema node specification.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            :exc:`dsch.schema.ValidationError`: if validation fails.
        """
        self.schema_node.validate(self.value)

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        node type, not on the selected storage backend.

        Returns:
            Node data.
        """
        raise NotImplementedError('To be implemented in subclass.')


class List:
    """List-type data node.

    :class:`List` is the base class for list-type data nodes, providing common
    functionality and the common interface. Subclasses may add functionality
    depending on the backend.

    Attributes:
        schema_node: The schema node that this data node is based on.
        parent: Parent data node object (``None`` if this is the top-level data
            node).
        empty: Data absence flag. ``True`` if no data is present.
    """

    def __init__(self, schema_node, parent, data_storage=None,
                 new_params=None):
        """Initialize list node from a given schema node.

        Note: Both ``data_storage`` and ``new_params`` are backend-specific.

        Args:
            schema_node: Schema node to create the data node for.
            parent: Parent data node object (``None`` if this is the top-level
                data node).
            data_storage: Backend-specific data storage object to load.
            new_params: Backend-specific metadata for data node creation.
        """
        self.schema_node = schema_node
        self.parent = parent
        self._subnodes = []
        if data_storage is not None:
            self._init_from_storage(data_storage)
        else:
            self._init_new(new_params)

    def append(self, value=None):
        """Append a new data node to the list.

        If a ``value`` is given, it is automatically applied to the new data
        node. Otherwise, an empty data node is created, which can be useful
        especially for Lists of Compilations.

        Note: This works with actual data values!

        Args:
            value: Value to be added to the list.
        """
        subnode = data_node_from_schema(self.schema_node.subnode,
                                        self.__module__, self)
        self._subnodes.append(subnode)
        if value is not None:
            subnode.replace(value)

    def clear(self):
        """Clear all subnodes."""
        self._subnodes.clear()

    @property
    def empty(self):
        """Check whether the List is currently empty.

        A List is considered empty when no sub-nodes are present.

        Returns:
            bool: ``True`` if the data node is empty, ``False`` otherwise.
        """
        return len(self._subnodes) == 0

    def __getitem__(self, item):
        """Return subnodes via the brackets syntax.

        This returns the entire sub-node object, not just the node value.
        """
        return self._subnodes[item]

    def __len__(self):
        """Return the length of the List, i.e. the number of subnodes."""
        return len(self._subnodes)

    def _init_from_storage(self, data_storage):
        """Initialize List from the given data storage object.

        This recursively initializes data nodes for all sub-nodes, using the
        given data storage object.

        The default implementation expects a dict, where keys are of the form
        "item_X", with X the list index, and values are the corresponding data
        storage objects. Backend-specific subclasses may change this, if
        required.

        Args:
            data_storage (dict): Backend-specific data storage object to load.
        """
        for _, node_storage in sorted(data_storage.items()):
            subnode = data_node_from_schema(self.schema_node.subnode,
                                            self.__module__, self,
                                            data_storage=node_storage)
            self._subnodes.append(subnode)

    def _init_new(self, new_params):
        """Initialize new, empty List data node.

        The default implementation does nothing. It is provided for interface
        consistency and for possible overriding by backend-specific subclasses.

        Args:
            new_params: Backend-specific metadata for data node creation.
        """
        pass

    def replace(self, new_value):
        """Replace the current list entries with the given list of entries.

        For :class:`List`, this is effectively a shorthand for calling
        :meth:`clear` and then, for each of the new entries, :meth:`append`.

        Args:
            new_value (list): New entries to put into the List.
        """
        self.clear()
        for item in new_value:
            self.append(item)

    def validate(self):
        """Recursively validate all sub-node values.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            :exc:`dsch.schema.ValidationError`: if validation fails.
        """
        self.schema_node.validate(self)
        for node in self._subnodes:
            node.validate()


def data_node_from_schema(schema_node, module_name, parent, data_storage=None,
                          new_params=None):
    """Create a new data node from a given schema node.

    Finds the data node class corresponding to the given schema node and
    creates an instance. However, the module containing the data node class
    must be given, which allows to select the desired storage backend.

    If ``data_storage`` is given, the new data node is initialized from that
    storage object. Otherwise, a new data node with a new storage object is
    created. Backends may use a ``new_params`` object to supply parameters
    for new data node creation.

    Args:
        schema_node: Schema node instance to create a data node for.
        module_name (str): The full module name of the data storage backend.
        parent: Parent data node object.
        data_storage: Backend-specific data storage object to load.
        new_params: Backend-specific metadata for data node creation.

    Returns:
        Data node corresponding to the given schema node.
    """
    backend_module = importlib.import_module(module_name)
    node_type_name = type(schema_node).__name__
    data_node_type = getattr(backend_module, node_type_name)
    return data_node_type(schema_node, parent, data_storage=data_storage,
                          new_params=new_params)
