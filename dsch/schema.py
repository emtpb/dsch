"""dsch schema specification.
"""


class Bool:
    """Schema node for scalar boolean values.

    This node type only accepts values of type `bool`.

    No configuration is required.
    """

    @classmethod
    def from_dict(cls, node_dict):
        """Create a new :class:`Bool` instance from a dict representation.

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
        if not type(test_data) == bool:
            raise ValidationError('Invalid type/value.', 'bool',
                                  type(test_data))


class Compilation:
    """Schema node for compound values composed from multiple named sub-nodes.

    Usually, a :class:`Compilation` is used to name and group related data.

    The data described by the compilation is expected to be given as an object
    with attributes corresponding to the sub-node names.
    While the functionality is more similar to a dict, the object
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
        """Create a new :class:`Compilation` instance from a dict
        representation.

        Args:
            node_dict: dict-representation of the node to be loaded.

        Returns:
            :class:`Compilation`: New compilation-type schema node.
        """
        if node_dict['node_type'] != 'Compilation':
            raise ValueError('Invalid node type in dict.')

        subnodes = {name: _node_from_dict(node_config) for name, node_config in
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

    def validate(self, test_data):
        """Validate given data against the node's constraints.

        For :class:`Compilation` nodes, this ensures that the ``test_data``
        is an object that has attributes according to the sub-node names.
        Then, the attribute values are individually (and recursively) validated
        through the corresponding sub-nodes.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Args:
            test_data: Data to be validated.

        Raises:
            :exc:`.ValidationError`: if validation fails.
        """
        for node_name, node in self.subnodes.items():
            try:
                data_value = getattr(test_data, node_name)
            except AttributeError:
                raise ValidationError('Missing data attribute.', node_name)
            node.validate(data_value)


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
        self.message = message
        self.expected = expected
        self.got = got


_node_types = {
    'Bool': Bool,
    'Compilation': Compilation,
}


def _node_from_dict(node_dict):
    """Create a new node from its ``node_dict``.

    This is effectively a shorthand for choosing the correct node class and
    then calling its ``from_dict`` method.

    Args:
        node_dict (dict): dict-representation of the node.

    Returns:
        New schema node with the specified type and configuration.
    """
    if node_dict['node_type'] not in _node_types:
        raise ValueError('Invalid node type specified.')
    node_type = _node_types[node_dict['node_type']]
    return node_type.from_dict(node_dict)
