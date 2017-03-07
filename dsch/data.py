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
    """

    def __init__(self, schema_node):
        """Initialize compilation node from a given schema node.

        Args:
            schema_node: Schema node to create the data node for.
        """
        self.schema_node = schema_node
        self._subnodes = {}
        for node_name, node in schema_node.subnodes.items():
            self._subnodes[node_name] = data_node_from_schema(
                node, self.__module__)

    def __dir__(self):
        """Include sub-nodes in :meth:`dir`."""
        attrs = super().__dir__()
        attrs.extend(self._subnodes.keys())
        return attrs

    def __getattr__(self, attr_name):
        """Return sub-nodes via the dot-attribute syntax.

        This returns the entire sub-node object, not just the node value.
        """
        return self._subnodes[attr_name]

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
        """Validate the subnode values against the schema node specification.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            :exc:`dsch.schema.ValidationError`: if validation fails.
        """
        self.schema_node.validate(self)


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
        storage: Backend-specific data storage object. This usually has a
            different type than the actual data value.
        value: Actual node data, independent of the backend in use.
    """

    def __init__(self, schema_node):
        """Initialize data node from a given schema node.

        Args:
            schema_node: Schema node to create the data node for.
        """
        self.schema_node = schema_node
        self.storage = None

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
    """

    def __init__(self, schema_node):
        """Initialize list node from a given schema node.

        Args:
            schema_node: Schema node to create the data node for.
        """
        self.schema_node = schema_node
        self._subnodes = []

    def append(self, value):
        """Append a new value to the list.

        Note: This works with actual data values!

        Args:
            value: Value to be added to the list.
        """
        subnode = data_node_from_schema(self.schema_node.subnode,
                                        self.__module__)
        subnode.replace(value)
        self._subnodes.append(subnode)

    def clear(self):
        """Clear all subnodes.
        """
        self._subnodes.clear()

    def __getitem__(self, item):
        """Return subnodes via the brackets syntax.

        This returns the entire sub-node object, not just the node value.
        """
        return self._subnodes[item]

    def __len__(self):
        """Return the length of the List, i.e. the number of subnodes."""
        return len(self._subnodes)

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
        """Validate the subnode values against the schema node specification.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            :exc:`dsch.schema.ValidationError`: if validation fails.
        """
        self.schema_node.validate(self)


def data_node_from_schema(schema_node, module_name):
    """Create a new data node from a given schema node.

    Finds the data node class corresponding to the given schema node and
    creates an instance. However, the module containing the data node class
    must be given, which allows to select the desired storage backend.

    Args:
        schema_node: Schema node instance to create a data node for.
        module_name (str): The full module name of the data storage backend.

    Returns:
        Data node corresponding to the given schema node.
    """
    backend_module = importlib.import_module(module_name)
    node_type_name = type(schema_node).__name__
    data_node_type = getattr(backend_module, node_type_name)
    return data_node_type(schema_node)
