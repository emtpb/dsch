"""dsch data representation.

In dsch, data is structured according to a given schema. The data is then
represented as a hierarchical structure of data nodes, each of which
corresponds to a node in the schema. This allows subsequent validation against
the schema.

The data nodes are also responsible for storing the data. Since dsch is built
to support multiple storage backends, there are specific data node classes
implementing the respective functionality. The classes in this module provide
common functionality and are intended to be used as base classes.
Different backends are implemented in the :mod:`dsch.backends` package.
"""
import datetime
import importlib
from . import schema


class ItemNode:
    """Generic data item node.

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
        complete: Data completeness flag. ``True`` if data is present.
        empty: Data absence flag. ``True`` if no data is present.
        value: Actual node data, independent of the backend in use.
    """

    def __init__(self, schema_node, parent, data_storage=None,
                 new_params=None):
        """Initialize data node from a given schema node.

        When ``data_storage`` is given, the data node is created from that
        storage object, i.e. loading existing data. Otherwise, a new data node
        is created, using optional ``new_params`` for creation, with an empty
        value.

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

    def clear(self):
        """Clear the data that is held by this data node.

        This removes the corresponding storage object entirely, causing the
        data node to be :attr:`empty` afterwards.
        """
        self._storage = None

    @property
    def complete(self):
        """Check whether the data node is currently complete.

        A data node is considered complete when a corresponding storage object
        exists. For non-containing nodes (i.e. all node types except
        :class:`Compilation` and :class:`List`), this is always the inverse of
        :attr:`empty`, but the property is still provided for interface
        compatibility.

        Returns:
            bool: ``True`` if the data node is complete, ``False`` otherwise.
        """
        return not self.empty

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
            :exc:`.schema.ValidationError`: if validation fails.
        """
        if self._storage is not None:
            self.schema_node.validate(self.value)

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        node type, not on the selected storage backend.

        If the node is currently empty, the value is undefined and
        :class:`NodeEmptyError` is raised.

        Returns:
            Node data.

        Raises:
            :exc:`NodeEmptyError`: if the node is currently empty.
        """
        if self._storage is None:
            raise NodeEmptyError()
        else:
            return self._value()

    def _value(self):
        """Return the actual node data, independent of the backend in use.

        Returns:
            Node data.
        """
        raise NotImplementedError('To be implemented in subclass.')


class Array(ItemNode):
    """Generic Array data node.

    This class implements backend-independent behaviour of Array data nodes.
    Backend-specific subclasses should derive from this class.
    """

    def __getitem__(self, key):
        """Pass slicing/indexing operations directly to NumPy array."""
        return self._storage[key]

    def resize(self, size):
        """Resize the array to the desired size.

        Args:
            size (tuple): Desired array size.
        """
        self._storage.resize(size)

    def __setitem__(self, key, value):
        """Pass slicing/indexing operations directly to NumPy array."""
        self._storage[key] = value

    def validate(self):
        """Validate the node value against the schema node specification.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            :exc:`.schema.ValidationError`: if validation fails.
        """
        if self._storage is None:
            return
        independent_values = []
        node_names = self.schema_node.depends_on or []
        for node_name in node_names:
            if node_name:
                independent_values.append(getattr(self.parent, node_name)
                                          .value)
        self.schema_node.validate(self.value, independent_values)


class Compilation:
    """Compilation data node.

    :class:`Compilation` is the base class for compilation-type data nodes,
    providing common functionality and the common interface. Subclasses may add
    functionality depending on the backend.

    Attributes:
        schema_node: The schema node that this data node is based on.
        parent: Parent data node object (``None`` if this is the top-level data
            node).
        complete: Data completeness flag. ``True`` if all required data is
            present.
        empty: Data absence flag. ``True`` if no data is present.
    """

    def __init__(self, schema_node, parent, data_storage=None,
                 new_params=None):
        """Initialize compilation node from a given schema node.

        When ``data_storage`` is given, the Compilation and all its sub-nodes
        are created from that data storage object, i.e. loading existing data.
        Otherwise, a new Compilation is created with empty sub-nodes, possibly
        using optional ``new_params`` for creation.
        The data node instances for the sub-nodes are recursively created in
        both cases.

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

    def clear(self):
        """Clear all sub-node values.

        Note that, in contrast to :class:`List`, this does not remove the
        the sub-nodes entirely, but only their values (by calling the
        respective :meth:`clear` method). This is because the set of sub-nodes
        for a Compilation is fixed via the schema specification and does not
        change during usage.
        """
        for node in self._subnodes.values():
            node.clear()

    @property
    def complete(self):
        """Check whether the Compilation is currently complete.

        A Compilation is considered complete when all non-optional sub-nodes
        are individually complete. This allows defining exceptions for specific
        sub-nodes by including them in
        :attr:`.schema.Compilation.optionals`.

        .. note::
            :attr:`complete` is not simply the inverse of :attr:`empty`, since
            it is only ``True`` when *all* non-optional fields are filled. This
            means a Compilation can be non-empty and non-complete at the same
            time.

        Returns:
            bool: ``True`` if the Compilation is complete, ``False`` otherwise.
        """
        required_nodes = [node for name, node in self._subnodes.items()
                          if name not in self.schema_node.optionals]
        for node in required_nodes:
            if not node.complete:
                return False
        return True

    def __dir__(self):
        """Include sub-nodes in :func:`dir`."""
        attrs = super().__dir__()
        attrs.extend(self._subnodes.keys())
        return attrs

    @property
    def empty(self):
        """Check whether the Compilation is currently empty.

        A Compilation is considered empty when all individual sub-nodes are
        empty.

        Returns:
            bool: ``True`` if the Compilation is empty, ``False`` otherwise.
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
            node_storage = data_storage.get(node_name, None)
            self._subnodes[node_name] = data_node_from_schema(
                subnode, self.__module__, self,
                data_storage=node_storage)

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
            :exc:`.data.SubnodeValidationError`: if validation fails.
        """
        for node_name, node in self._subnodes.items():
            try:
                node.validate()
            except (schema.ValidationError, SubnodeValidationError) as err:
                raise SubnodeValidationError(node_name) from err


class Date(ItemNode):
    """Generic Date data node.

    This class implements backend-independent behaviour of Date data nodes.
    Backend-specific subclasses should derive from this class.
    """

    def _init_new(self, new_params):
        """Initialize new Date data node.

        If the corresponding schema node's
        :attr:`.schema.Date.set_on_create` is ``True``, the data node's
        value is automatically initialized with the current date. Otherwise,
        it is left empty and can be filled by calling meth:`replace`.

        Args:
            new_params: Backend-specific metadata for data node creation.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.replace(datetime.date.today())


class DateTime(ItemNode):
    """Generic DateTime data node.

    This class implements backend-independent behaviour of DateTime data nodes.
    Backend-specific subclasses should derive from this class.
    """

    def _init_new(self, new_params):
        """Initialize new DateTime data node.

        If the corresponding schema node's
        :attr:`.schema.DateTime.set_on_create` is ``True``, the data node's
        value is automatically initialized with the current date and time.
        Otherwise, it is left empty and can be filled by calling
        meth:`replace`.

        Args:
            new_params: Backend-specific metadata for data node creation.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.replace(datetime.datetime.now())


class List:
    """List-type data node.

    :class:`List` is the base class for list-type data nodes, providing common
    functionality and the common interface. Subclasses may add functionality
    depending on the backend.

    Attributes:
        schema_node: The schema node that this data node is based on.
        parent: Parent data node object (``None`` if this is the top-level data
            node).
        complete: Data completeness flag. ``True`` if all list items are
            complete.
        empty: Data absence flag. ``True`` if no data is present.
    """

    def __init__(self, schema_node, parent, data_storage=None,
                 new_params=None):
        """Initialize list node from a given schema node.

        When ``data_storage`` is given, the List and all its sub-nodes are
        created from that data storage object, i.e. loading existing data.
        Otherwise, a new List is created without sub-nodes, possibly
        using optional ``new_params`` for creation.

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

        Args:
            value: Value to be added to the list.
        """
        subnode = data_node_from_schema(self.schema_node.subnode,
                                        self.__module__, self)
        self._subnodes.append(subnode)
        if value is not None:
            subnode.replace(value)

    def clear(self):
        """Clear all sub-nodes.

        This removes all sub-nodes entirely, yielding an empty List.
        """
        self._subnodes.clear()

    @property
    def complete(self):
        """Check whether the List is currently complete.

        A List is considered complete when all of its sub-nodes are complete.

        .. warning::
            An empty List is considered complete! If a minimum number of list
            items is required, use :attr:`.schema.List.min_length` to apply
            the corresponding constraint.

        Returns:
            bool: ``True`` if the List is complete, ``False`` otherwise.
        """
        for node in self._subnodes:
            if not node.complete:
                return False
        return True

    @property
    def empty(self):
        """Check whether the List is currently empty.

        A List is considered empty when no sub-nodes are present.

        Returns:
            bool: ``True`` if the List is empty, ``False`` otherwise.
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
            :exc:`.data.SubnodeValidationError`: if validation fails.
        """
        self.schema_node.validate(self)
        for idx, node in enumerate(self._subnodes):
            try:
                node.validate()
            except (schema.ValidationError, SubnodeValidationError) as err:
                raise SubnodeValidationError(idx) from err


class Time(ItemNode):
    """Generic Time data node.

    This class implements backend-independent behaviour of Time data nodes.
    Backend-specific subclasses should derive from this class.
    """

    def _init_new(self, new_params):
        """Initialize new Time data node.

        For :class:`Time`, the corresponding HDF5 dataset is only created if
        the corresponding schema node's :attr:`.schema.Time.set_on_create` is
        ``True``. Otherwise, the dataset is left empty and can be filled by
        calling meth:`replace`.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.replace(datetime.datetime.now().time())


class NodeEmptyError(Exception):
    """Exception indicating an empty data node.

    This exception is raised when requesting a data node's value fails because
    the node is empty. In that case, the value is undefined.
    """

    def __init__(self):
        super().__init__('Node is empty. The value of empty nodes is '
                         'undefined.')


class SubnodeValidationError(Exception):
    """Exception used when validation fails for a subnode.

    Validation failure always raises :class:`.schema.ValidationError`. However,
    this only indicates which check has failed (e.g. string length exceeded),
    but not for which field this has occured.

    This class uses exception chaining to recursively determine the full name
    and path of the affected data node, which is provided via
    :meth:`node_path`.
    """

    def __init__(self, location):
        """Initialize SubnodeValidationError instance.

        The given parameter identifies the "inner" node (causing the exception)
        from the scope of the "outer" node (e.g. a :class:`Compilation` or
        :class:`List`).

        For example, for a :class:`String` node inside a :class:`Compilation`,
        the string is the inner node, failing validation, and the compilation
        is the outer node, defining the name of the string node. In this case,
        the compilation would set ``location`` to the name of the string node.
        Similarly, for :class:`List` nodes as outer nodes, ``location`` is set
        to the corresponding list item's index.

        :class:`SubnodeValidationError` can also be created from other
        instances of themselves, thus representing nested structures of lists
        and compilations.

        Args:
            location (:class:`str` or :class:`int`): Node location info.
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
        elif isinstance(cause, schema.ValidationError):
            post = ''

        if isinstance(self._location, str):
            fmt = '{loc}{post}'
        elif isinstance(self._location, int):
            fmt = '[{loc}]{post}'
        return fmt.format(loc=self._location, post=post)


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
