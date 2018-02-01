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

import asciitree

from . import exceptions

draw_tree = asciitree.LeftAligned(
    draw=asciitree.BoxStyle(gfx=asciitree.drawing.BOX_LIGHT, horiz_len=1)
)


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
        exists. For applying a new value, set :attr:`value`.

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

    def load_from(self, source_node):
        """Load data by copying from the given source node.

        This is effectively a shorthand for ``self.replace(source_node.value)``
        with additional checking of node compatibility. Two nodes are
        considered compatible if their ``schema_node`` attributes are
        identical.

        Args:
            source_node: Data node to copy value from.
        """
        if source_node.schema_node.hash() != self.schema_node.hash():
            raise exceptions.IncompatibleNodesError(
                source_node.schema_node.hash(), self.schema_node.hash())
        self.replace(source_node.value)

    def node_tree(self):
        """Return a recursive representation of the (sub)node-tree.

        The representation is a dict with the node's own label as the key and
        the tree of sub-nodes as the value. The label always starts with the
        node type in parentheses.

        For leaf nodes, i.e. nodes that do not contain other nodes, the
        :func:`str`-representation of the value is also included in the label.
        If no value is set, '<empty>' is printed instead.

        Returns:
            dict: {label: sub_tree} representation.
        """
        try:
            value = str(self.value)
        except exceptions.NodeEmptyError:
            value = '<empty>'
        key = '({type_name}): {value}'.format(type_name=type(self).__name__,
                                              value=value)
        return {key: {}}

    def replace(self, new_value):
        """Completely replace the current node value.

        Instead of changing parts of the data (e.g. via numpy array slicing),
        replace the entire data object for this node.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        raise NotImplementedError('To be implemented in subclass.')

    def __repr__(self):
        return draw_tree(self.node_tree())

    def validate(self):
        """Validate the node value against the schema node specification.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            dsch.exceptions.ValidationError: if validation fails.
        """
        if self._storage is not None:
            self.schema_node.validate(self.value)

    @property
    def value(self):
        """Return the actual node data, independent of the backend in use.

        This representation of the data only depends on the corresponding
        node type, not on the selected storage backend.

        If the node is currently empty, the value is undefined and
        :exc:`~dsch.exceptions.NodeEmptyError` is raised.

        Returns:
            Node data.

        Raises:
            dsch.exceptions.NodeEmptyError: if the node is currently empty.
        """
        if self._storage is None:
            raise exceptions.NodeEmptyError()
        else:
            return self._value()

    @value.setter
    def value(self, new_value):
        """Apply a new value to the data node.

        This is a convenience shortcut to :meth:`replace`.

        Args:
            new_value: New value to apply to the node, independent of the
                backend in use.
        """
        self.replace(new_value)

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

    def node_tree(self):
        """Return a recursive representation of the (sub)node-tree.

        The representation is a dict with the node's own label as the key and
        the tree of sub-nodes as the value. The label always starts with the
        node type in parentheses.

        For Array nodes, the value is *not* included in the label because of
        its length. Instead, the array shape is shown. If no value is set,
        '<empty>' is printed instead.

        Returns:
            dict: {label: sub_tree} representation.
        """
        try:
            value = 'x'.join([str(l) for l in self.value.shape]) + ' array'
        except exceptions.NodeEmptyError:
            value = '<empty>'
        key = '({type_name}): {value}'.format(type_name=type(self).__name__,
                                              value=value)
        return {key: {}}

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
            dsch.exceptions.ValidationError: if validation fails.
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

    def load_from(self, source_node):
        """Load data by copying from the given source node.

        For Compilations, this copies the relevant subnode's data recursively.

        Args:
            source_node: Data node to copy value from.
        """
        if source_node.schema_node.hash() != self.schema_node.hash():
            raise exceptions.IncompatibleNodesError(
                source_node.schema_node.hash(), self.schema_node.hash())
        for key, subnode in self._subnodes.items():
            subnode.load_from(getattr(source_node, key))

    def node_tree(self):
        """Return a recursive representation of the (sub)node-tree.

        The representation is a dict with the node's own label as the key and
        the tree of sub-nodes as the value. The label always starts with the
        node type in parentheses.

        For Compilation nodes, all sub-node's representations are printed
        recursively, prefixed by the sub-node name.

        Returns:
            dict: {label: sub_tree} representation.
        """
        tree = {}
        for name, subnode in self._subnodes.items():
            for sub_label, sub_tree in subnode.node_tree().items():
                key = '{name} {label}'.format(name=name, label=sub_label)
                tree[key] = sub_tree
        return {'(Compilation)': tree}

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

    def __repr__(self):
        return draw_tree(self.node_tree())

    def __setattr__(self, attr_name, new_value):
        """Prevent accidental (re-)setting of sub-nodes.

        When the user wishes to change a value of a Compilation's sub-node,
        they must access something like ``comp.sub.value``. If the ``value`` is
        omitted or forgotten, a write operation would replace the entire data
        node with the user's value, breaking dsch behaviour in various places.

        To prevent this, `setattr` operations are only permitted for the
        Compilation's few actual attributes. For everything else, a
        :exc:`~dsch.exceptions.ResetSubnodeError` is raised, mentioning the
        ``value`` attribute.

        Raises:
            dsch.exceptions.ResetSubnodeError: if attempting to `set` a
                sub-node.
        """
        if attr_name not in ('schema_node', 'parent', '_subnodes'):
            raise exceptions.ResetSubnodeError(attr_name)
        super().__setattr__(attr_name, new_value)

    def validate(self):
        """Recursively validate all sub-node values.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            dsch.exceptions.SubnodeValidationError: if validation fails.
        """
        for node_name, node in self._subnodes.items():
            try:
                node.validate()
            except (exceptions.ValidationError,
                    exceptions.SubnodeValidationError) as err:
                raise exceptions.SubnodeValidationError(node_name) from err


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
        it is left empty and can be filled by setting :attr:`value`.

        Args:
            new_params: Backend-specific metadata for data node creation.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.value = datetime.date.today()


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
        Otherwise, it is left empty and can be filled by setting :attr:`value`.

        Args:
            new_params: Backend-specific metadata for data node creation.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.value = datetime.datetime.now()


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

        A List is considered empty when all of its sub-nodes are empty. As a
        special case, it is also considered empty when there are no sub-nodes
        present.

        Returns:
            bool: ``True`` if the List is empty, ``False`` otherwise.
        """
        for subnode in self._subnodes:
            if not subnode.empty:
                return False
        return True

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
        for idx in range(len(data_storage)):
            node_storage = data_storage['item_{}'.format(idx)]
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

    def load_from(self, source_node):
        """Load data by copying from the given source node.

        For Lists, this copies the relevant subnode's data recursively.

        Args:
            source_node: Data node to copy value from.
        """
        if source_node.schema_node.hash() != self.schema_node.hash():
            raise exceptions.IncompatibleNodesError(
                source_node.schema_node.hash(), self.schema_node.hash())
        for idx, subnode in enumerate(source_node):
            self.append()
            self[idx].load_from(subnode)

    def node_tree(self):
        """Return a recursive representation of the (sub)node-tree.

        The representation is a dict with the node's own label as the key and
        the tree of sub-nodes as the value. The label always starts with the
        node type in parentheses.

        For lists, all sub-node's representations are printed recursively,
        prefixed by the list index in brackets.

        Returns:
            dict: {label: sub_tree} representation.
        """
        tree = {}
        for idx, subnode in enumerate(self._subnodes):
            for sub_label, sub_tree in subnode.node_tree().items():
                key = '[{idx}] {label}'.format(idx=idx, label=sub_label)
                tree[key] = sub_tree
        return {'(List)': tree}

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

    def __repr__(self):
        return draw_tree(self.node_tree())

    def __setitem__(self, idx, new_value):
        """Prevent accidental (re-)setting of items.

        When the user wishes to change a value of a List item, they must access
        something like ``lst[0].value``. If the ``value`` is omitted or
        forgotten, a write operation would replace the entire data node with
        the user's value, breaking dsch behaviour in various places.

        Therefore, `setitem` operations are generally not permitted. If
        attempted, a :exc:`~dsch.exceptions.ResetSubnodeError` is raised,
        mentioning the ``value`` attribute.

        Raises:
            dsch.exceptions.ResetSubnodeError: if attempting to `set` a list
                item.
        """
        raise exceptions.ResetSubnodeError('[{}]'.format(idx))

    def validate(self):
        """Recursively validate all sub-node values.

        If validation succeeds, the method terminates silently. Otherwise, an
        exception is raised.

        Raises:
            dsch.exceptions.SubnodeValidationError: if validation fails.
        """
        self.schema_node.validate(self)
        for idx, node in enumerate(self._subnodes):
            try:
                node.validate()
            except (exceptions.ValidationError,
                    exceptions.SubnodeValidationError) as err:
                raise exceptions.SubnodeValidationError(idx) from err


class Scalar(ItemNode):
    """Generic Scalar data node.

    This class implements backend-independent behaviour of Scalar data nodes.
    Backend-specific subclasses should derive from this class.
    """

    def node_tree(self):
        """Return a recursive representation of the (sub)node-tree.

        The representation is a dict with the node's own label as the key and
        the tree of sub-nodes as the value. The label always starts with the
        node type in parentheses.

        For Scalar nodes, the :attr:`~dsch.schema.Scalar.unit` is appended to
        the value, if any. If no value is set, '<empty>' is printed instead.

        Returns:
            dict: {label: sub_tree} representation.
        """
        try:
            value = '{value} {unit}'.format(value=self.value,
                                            unit=self.schema_node.unit)
        except exceptions.NodeEmptyError:
            value = '<empty>'
        key = '({type_name}): {value}'.format(type_name=type(self).__name__,
                                              value=value)
        return {key: {}}


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
        setting :attr:`value`.

        Args:
            new_params (dict): Dict including the HDF5 dataset name as ``name``
                and the HDF5 parent object as ``parent``.
        """
        super()._init_new(new_params)
        if self.schema_node.set_on_create:
            self.value = datetime.datetime.now().time()


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
