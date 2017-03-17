"""dsch schema specification.

In dsch, data is structured according to a given schema, which must be defined
prior to working with the data (i.e. saving and loading). A schema is a
tree-like hierarchical structure of schema nodes, each of which applies certain
constraints to the data. In general, there are three different kinds of schema
nodes:

1. Items
    represent a data point (e.g. a string or a NumPy array). These are the
    leaves of the tree.
2. Compilations
    represent a compound of data made up from multiple named fields.
    Each field is represented by another node and therefore supports its own
    constraints.
3. Lists
    can contain multiple elements of the same type. The constraints are
    described with a single sub-node, but then applied to all data elements.

Compilations and lists both support all kinds of sub-nodes and can be nested.
This allows to specify arbitrary hierarchically structured schemas.

.. note::
    The classes in this module are used to specify a schema. This specification
    is different from what the user sees when interacting with the actual data.
    For example, a list-type *schema node* only has a single sub-node, since
    the schema node only specifies the constraints to be applied to the data.
    When users interact with the data, however, they use *data nodes*, which,
    in the case of lists, can contain multiple sub-nodes.
    Data nodes are defined in the :mod:`dsch.data` module.
"""
import numpy as np


class Array:
    """Schema node for NumPy ndarray values.

    This node type accepts values of type :class:`numpy.ndarray`.

    In addition to the actual value(s), Arrays contain metadata:

    * The unit of the physical quantity that is represented by the values,
      e.g. 'V' for volts.

    Also, Arrays support various constraints:

    * NumPy data type (:class:`numpy.dtype`). This directly validates the
      array's ``dtype``. Note that the data type is always matched exactly, so
      one cannot require "any of the int-dtypes".
      This attribute is non-optional, since many backends require knowledge of
      the data type for efficient storage.
    * Minimum and maximum array shape. Like :attr:`numpy.ndarray.shape`, this
      tuples limit the size of each dimension of the array. If both minimum and
      maximum shape are given, they must have the same length.
    * Number of array dimensions. If a minimum or maximum array shape is given,
      this is determined automatically. Otherwise, it can be given explicitly
      or left at the default value of ``1``.
    * Minimum and maximum array values. This constraint is applied to each
      individual value in the array, meaning that a single array element value
      outside the given boundaries causes validation to fail.

    Note: The constraint parameters for array shape and values all default to
    ``None``, effectively disabling the corresponding validation step.

    Attributes:
        dtype (:class:`numpy.dtype` or :class:`str`): Required NumPy dtype.
        unit (str): Unit of the physical quantity, e.g. 'V' for volts. Unit
            (and values) should be given without any SI prefixes.
        max_shape (tuple): Maximum allowed array shape.
        min_shape (tuple): Minimum allowed array shape.
        ndim (int): Number of array dimensions.
        max_value: Maximum allowed value for any element.
        min_value: Minimum allowed value for any element.
    """

    def __init__(self, dtype, unit='', max_shape=None, min_shape=None,
                 ndim=1, max_value=None, min_value=None):
        """Initialize array-type schema node.

        Args:
            dtype (:class:`numpy.dtype` or str): Required NumPy dtype.
            unit (str): Unit of the physical quantity, e.g. 'V' for volts. Unit
                (and values) should be given without any SI prefixes.
            max_shape (tuple): Maximum allowed array shape.
            min_shape (tuple): Minimum allowed array shape.
            ndim (int): Number of array dimensions.
            max_value: Maximum allowed value for any element.
            min_value: Minimum allowed value for any element.
        """
        if max_shape and min_shape and len(max_shape) != len(min_shape):
            raise ValueError('Shape constraints must have the same length.')

        self.dtype = dtype
        self.unit = unit
        self.max_shape = max_shape
        self.min_shape = min_shape
        if max_shape or min_shape:
            self.ndim = len(max_shape or min_shape)
        else:
            self.ndim = ndim
        self.max_value = max_value
        self.min_value = min_value

    @classmethod
    def from_dict(cls, node_dict):
        """Create a new instance from a dict representation.

        Args:
            node_dict: dict-representation of the node to be loaded.

        Returns:
            :class:`Array`: New array-type schema node.
        """
        if node_dict['node_type'] != 'Array':
            raise ValueError('Invalid node type in dict.')
        return cls(**node_dict['config'])

    def to_dict(self):
        """Return the node representation as a dict.

        The representation dict includes a field ``node_type`` with the node
        class name and a field ``config`` with a dict of the configuration
        options.

        Returns:
            dict: dict-representation of the node.
        """
        config = {
            'unit': self.unit,
            'max_shape': self.max_shape,
            'min_shape': self.min_shape,
            'ndim': self.ndim,
            'max_value': self.max_value,
            'min_value': self.min_value,
            'dtype': self.dtype,
        }
        return {'node_type': 'Array', 'config': config}

    def validate(self, test_data):
        """Validate given data against the node's constraints.

        For :class:`Array` nodes, this ensures that the given data type is of
        type :class:`numpy.ndarray` and that all constraints (dimensions, dtype
        etc.) are met.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Args:
            test_data: Data to be validated.

        Raises:
            :exc:`.ValidationError`: if validation fails.
        """
        if type(test_data) != np.ndarray:
            raise ValidationError('Invalid type/value.', 'numpy.ndarray',
                                  type(test_data))
        if test_data.ndim != self.ndim:
            raise ValidationError('Invalid number of array dimensions.',
                                  self.ndim, test_data.ndim)
        if self.max_shape:
            # Validate individual dimension's sizes
            for index, value in enumerate(test_data.shape):
                if self.max_shape[index] and value > self.max_shape[index]:
                    raise ValidationError('Maximum array shape exceeded.',
                                          self.max_shape, test_data.shape)
        if self.min_shape:
            # Validate individual dimension's sizes
            for index, value in enumerate(test_data.shape):
                if self.min_shape[index] and value < self.min_shape[index]:
                    raise ValidationError('Minimum array shape undercut.',
                                          self.min_shape, test_data.shape)
        if self.max_value is not None and np.any(test_data > self.max_value):
            raise ValidationError('Maximum array element value exceeded.',
                                  self.max_value,
                                  test_data[test_data > self.max_value])
        if self.min_value is not None and np.any(test_data < self.min_value):
            raise ValidationError('Minimum array element value undercut.',
                                  self.min_value,
                                  test_data[test_data < self.min_value])
        if self.dtype and not test_data.dtype == self.dtype:
            raise ValidationError('Invalid dtype.', self.dtype,
                                  test_data.dtype)


class Bool:
    """Schema node for scalar boolean values.

    This node type only accepts values of type :class:`bool`.

    No configuration is required.
    """

    @classmethod
    def from_dict(cls, node_dict):
        """Create a new instance from a dict representation.

        Args:
            node_dict: dict-representation of the node to be loaded.

        Returns:
            :class:`Bool`: New bool-type schema node.
        """
        if node_dict['node_type'] != 'Bool':
            raise ValueError('Invalid node type in dict.')
        return cls(**node_dict['config'])

    def to_dict(self):
        """Return the node representation as a dict.

        The representation dict includes a field ``node_type`` with the node
        class name and a field ``config`` with a dict of the configuration
        options.

        Returns:
            dict: dict-representation of the node.
        """
        return {'node_type': 'Bool', 'config': {}}

    def validate(self, test_data):
        """Validate given data against the node's constraints.

        For :class:`Bool` nodes, this ensures that the given data type is of
        type :class:`bool`.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Args:
            test_data: Data to be validated.

        Raises:
            :exc:`.ValidationError`: if validation fails.
        """
        if type(test_data) != bool:
            raise ValidationError('Invalid type/value.', 'bool',
                                  type(test_data))


class Compilation:
    """Schema node for compound values composed from multiple named sub-nodes.

    Usually, a :class:`Compilation` is used to name and group related data.
    Together with :class:`List`, this node type allows to build arbitrary
    hierarchical schemas.

    The corresponding data node is a subclass of :class:`dsch.data.Compilation`
    and provides attributes corresponding to the schema node's sub-node names.
    Each of those attributes then represents a data node.

    While the general functionality is more similar to a dict, the object
    representation is preferred because of the more compact "dotted" notation.
    This is especially relevant when nesting compilations, e.g.
    ``measurement.sampling.frequency`` vs.
    ``measurement['sampling']['frequency']``.

    Attributes:
        subnodes: dict-like mapping of names to schema sub-nodes.
    """

    def __init__(self, subnodes):
        """Initialize a Compilation schema node.

        Args:
            subnodes (dict): dict-like mapping of names to sub-nodes.
        """
        self.subnodes = subnodes

    @classmethod
    def from_dict(cls, node_dict):
        """Create a new instance from a dict representation.

        Args:
            node_dict: dict-representation of the node to be loaded.

        Returns:
            :class:`Compilation`: New compilation-type schema node.
        """
        if node_dict['node_type'] != 'Compilation':
            raise ValueError('Invalid node type in dict.')

        subnodes = {name: node_from_dict(node_config) for name, node_config in
                    node_dict['config']['subnodes'].items()}
        return cls(subnodes=subnodes)

    def to_dict(self):
        """Return the node representation as a dict.

        The representation dict includes a field ``node_type`` with the node
        class name and a field ``config`` with a dict of the configuration
        options.

        Returns:
            dict: dict-representation of the node.
        """
        subnode_dict = {name: node.to_dict() for name, node in
                        self.subnodes.items()}
        return {'node_type': 'Compilation', 'config': {
            'subnodes': subnode_dict
        }}


class List:
    """Schema node for lists of same-type elements.

    A :class:`List` is used to represent multiple data items that must meet the
    same constraints. It uses a single schema node to specify the constraints
    for all entries.
    Note that this behavior is different from regular python lists, which can
    contain arbitrary entries.

    Often, a :class:`List` is used with a :class:`Compilation` as its sub-node,
    allowing to represent arbitrary hierarchical schemas.

    Attributes:
        subnode: A single schema node, used to validate all list entries.
    """

    def __init__(self, subnode):
        """Initialize a List schema node.

        Args:
            subnode: A single schema node.
        """
        self.subnode = subnode

    @classmethod
    def from_dict(cls, node_dict):
        """Create a new instance from a dict representation.

        Args:
            node_dict: dict-representation of the node to be loaded.

        Returns:
            :class:`List`: New list-type schema node.
        """
        if node_dict['node_type'] != 'List':
            raise ValueError('Invalid node type in dict.')

        subnode = node_from_dict(node_dict['config']['subnode'])
        return cls(subnode)

    def to_dict(self):
        """Return the node representation as a dict.

        The representation dict includes a field ``node_type`` with the node
        class name and a field ``config`` with a dict of the configuration
        options.

        Returns:
            dict: dict-representation of the node.
        """
        node_dict = {'node_type': 'List', 'config': {
            'subnode': self.subnode.to_dict()
        }}
        return node_dict


class String:
    """Schema node for string values.

    This node type accepts regular Python strings, i.e. :class:`str` objects.
    Constraints can be optionally configured for minimum and maximum string
    length.

    Attributes:
        min_length (int): Minimum allowed string length.
        max_length (int): Maximum allowed string length.
    """

    def __init__(self, min_length=None, max_length=None):
        """Initialize string-type schema node.

        Args:
            min_length (int): Minimum allowed string length.
            max_length (int): Maximum allowed string length.
        """
        self.min_length = min_length
        self.max_length = max_length

    @classmethod
    def from_dict(cls, node_dict):
        """Create a new instance from a dict representation.

        Args:
            node_dict: dict-representation of the node to be loaded.

        Returns:
            :class:`String`: New string-type schema node.
        """
        if node_dict['node_type'] != 'String':
            raise ValueError('Invalid node type in dict.')
        return cls(**node_dict['config'])

    def to_dict(self):
        """Return the node representation as a dict.

        The representation dict includes a field ``node_type`` with the node
        class name and a field ``config`` with a dict of the configuration
        options.

        Returns:
            dict: dict-representation of the node.
        """
        config = {'min_length': self.min_length, 'max_length': self.max_length}
        return {'node_type': 'String', 'config': config}

    def validate(self, test_data):
        """Validate given data against the node's constraints.

        For :class:`String` nodes, this ensures that the given data type is of
        type :class:`str`, and that the string length is within the limits.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Args:
            test_data: Data to be validated.

        Raises:
            :exc:`.ValidationError`: if validation fails.
        """
        if type(test_data) != str:
            raise ValidationError('Invalid type/value.', 'str',
                                  type(test_data))
        if self.max_length and len(test_data) > self.max_length:
            raise ValidationError('Maximum string length exceeded.',
                                  self.max_length, len(test_data))
        if self.min_length and len(test_data) < self.min_length:
            raise ValidationError('Minimum string length undercut.',
                                  self.min_length, len(test_data))


class ValidationError(Exception):
    """Exception used when a schema node's validation fails.

    The :attr:`message` should indicate the kind of validation error and
    mention the constraint that was not met (e.g. 'Maximum length exceeded.').
    This message may be directly shown to users.
    For more information, :attr:`expected` contains a representation of
    the constraint (e.g. the maximum valid length) and :attr:`got` is the
    actual value determined for the given data, if applicable.

    Attributes:
        message (str): Human-readable error message.
        expected: Expected, i.e. valid value.
        got: Actual, i.e. invalid value.
    """

    def __init__(self, message, expected, got=None):
        """Initialize ValidationError instance."""
        super().__init__()
        self.message = message
        self.expected = expected
        self.got = got


def node_from_dict(node_dict):
    """Create a new node from its ``node_dict``.

    This is effectively a shorthand for choosing the correct node class and
    then calling its ``from_dict`` method.

    Args:
        node_dict (dict): dict-representation of the node.

    Returns:
        New schema node with the specified type and configuration.
    """
    node_types = {name: cls for name, cls in globals().items()
                  if not name.startswith('_') and
                  name not in ('ValidationError', 'node_from_dict')}
    if node_dict['node_type'] not in node_types:
        raise ValueError('Invalid node type specified.')
    node_type = node_types[node_dict['node_type']]
    return node_type.from_dict(node_dict)
