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
