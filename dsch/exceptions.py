"""Custom exception classes used throughout dsch."""

class DschError(Exception):
    """Base class for custom exceptions raised by dsch.

    All dsch-specific exceptions are derived from this class.
    """

    pass


class AutodetectBackendError(DschError):
    """Exception raised when backend autodetection fails."""

    def __init__(self, storage_path):
        """Initialize AutodetectBackendError.

        Args:
            storage_path: Storage path that auto-detection failed for.
        """
        super().__init__('Coult not auto-detect backend for storage "{}".'
                         .format(storage_path))


class NodeEmptyError(DschError):
    """Exception indicating an empty data node.

    This exception is raised when requesting a data node's value fails because
    the node is empty. In that case, the value is undefined.
    """

    def __init__(self):
        """Initialize new NodeEmptyError instance."""
        super().__init__('Node is empty. The value of empty nodes is '
                         'undefined.')


class IncompatibleNodesError(DschError):
    """Exception used when node operations fail due to incompatible schemas."""

    def __init__(self, hash1, hash2):
        """Initilize IncompatibleNodesError.

        Args:
            hash1 (str): Schema hash corresponding to the first node.
            hash2 (str): Schema hash corresponding to the second node.
        """
        msg = 'Nodes with schema hashes {} and {} are incompatible.'.format(
            hash1, hash2)
        super().__init__(msg)


class SubnodeValidationError(DschError):
    """Exception used when validation fails for a subnode.

    Validation failure always raises :exc:`ValidationError`. However, this only
    indicates which check has failed (e.g. string length exceeded), but not for
    which field this has occured.

    This class uses exception chaining to recursively determine the full name
    and path of the affected data node, which is provided via
    :meth:`node_path`.
    """

    def __init__(self, location):
        """Initialize SubnodeValidationError instance.

        The given parameter identifies the "inner" node (causing the exception)
        from the scope of the "outer" node (e.g. a
        :class:`~dsch.data.Compilation` or :class:`~dsch.data.List`).

        For example, for an :class:`~dsch.data.Array` node inside a
        :class:`~dsch.data.Compilation`, the Array is the inner node, failing
        validation, and the Compilation is the outer node, defining the name of
        the Array node. In this case, the Compilation would set ``location`` to
        the name of the Array node. Similarly, for :class:`List` nodes as outer
        nodes, ``location`` is set to the corresponding list item's index.

        :exc:`SubnodeValidationError` can also be created from other instances
        of themselves, thus representing nested structures of lists and
        compilations.

        Args:
            location (str or int): Node location info.
        """
        super().__init__()
        self._location = location

    def node_path(self):
        """Recursively determine the path of the node failing validation.

        Returns:
            str: Full name and path of the node that failed validation.
        """
        cause = self.__cause__
        if isinstance(cause, SubnodeValidationError):
            cause_path = cause.node_path()
            if cause_path.startswith('['):
                post = cause_path
            else:
                post = '.' + cause_path
        elif isinstance(cause, ValidationError):
            post = ''

        if isinstance(self._location, str):
            fmt = '{loc}{post}'
        elif isinstance(self._location, int):
            fmt = '[{loc}]{post}'
        return fmt.format(loc=self._location, post=post)

    def original_cause(self):
        """Get the exception originally causing the chain.

        This recursively follows the exception chain back to the original
        :exc:`ValidationError` that further describes the problem.

        Returns:
            :exc:`ValidationError`: Original cause exception.
        """
        if isinstance(self.__cause__, SubnodeValidationError):
            return self.__cause__.original_cause()
        elif isinstance(self.__cause__, ValidationError):
            return self.__cause__

    def __str__(self):
        """Return a nicely printable string representation."""
        return 'Node "{node_path}" failed validation: {msg}'.format(
            node_path=self.node_path(), msg=self.original_cause())


class ValidationError(DschError):
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

    def __str__(self):
        """Return a nicely printable string representation."""
        return '{msg} (Expected: {exp}. Got: {got})'.format(
            msg=self.message, exp=self.expected, got=self.got)
