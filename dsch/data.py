"""dsch data representation.

In dsch, data is structured according to a given schema. The data is then
represented as a hierarchical structure of data nodes, each of which
corresponds to a node in the schema. This allows subsequent validation against
the schema.

The data nodes are also responsible for storing the data. Since dsch is built
to support multiple storage backends, there are specific data node classes
implementing the respective functionality. The classes in this module provide
common functionality and are intended to be used as base classes.
"""
import importlib


class ItemNode:
    """Generic data node.

    :class:`ItemNode` is the base class for data nodes, providing common
    functionality and the common interface. Subclasses may add functionality
    depending on the node type and backend (e.g. compression settings).

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
