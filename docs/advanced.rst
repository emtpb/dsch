.. _advanced:

***************
Advanced Topics
***************


Schema design
=============

Nested schemas
--------------

The example schema from the :ref:`Tutorial <tutorial>` is a very simple one, so
let's extend it!
Suppose we do not only want to store a single measurement result of our weather
station, but multiple results taken at different times.
A simple solution for this would be to wrap the previous schema in a
:class:`~dsch.schema.List`::

   schema_list = dsch.schema.List(
       dsch.schema.Compilation({
           'time': dsch.schema.DateTime(),
           'temperature': dsch.schema.Scalar(dtype='float', unit='°C'),
           'humidity': dsch.schema.Scalar(dtype='float', unit='%', min_value=0,
                                          max_value=100)
       })
   )

If we now create a storage for this schema, we can use
:meth:`~dsch.data.List.append` to add individual measurement results to the
list::

   storage_list = dsch.create('list.h5', schema_list)
   storage_list.data.append({
       'time': datetime.datetime.now(),
       'temperature': 21,
       'humidity': 42
   })

Each list item is a Compilation that behaves exactly like in the Tutorial::

   >>> storage_list.data[0].temperature.value
   21.0

Alternatively, if you prefer to work with the data nodes directly, the argument
to ``append`` can be omitted, creating an empty data node::

   storage_list.data.append()
   storage_list.data[0].time.value = datetime.datetime.now()
   storage_list.data[0].temperature.value = 21
   storage_list.data[0].humidity.value = 42

By nesting Lists and Compilations, arbitrary schemas can be composed.

Schema extension
----------------

When working with measurement devices, a common approach is to implement small
controller libraries for every device model. These libraries can be used by an
application to control multiple devices in a measurement system, and to
aggregate their data.

Dsch allows every library to define a schema for its result datasets. The
application can then define another schema, incorporating all required library
schemas and possibly adding other fields::

   from lib1 import schema as schema_lib1
   from lib2 import schema as schema_lib2
   schema_app = dsch.schema.Compilation({
       'lib1': schema_lib1,
       'lib2': schema_lib2,
       'app_data': dsch.schema.Compilation(...),
   })

This way, the original library schema is preserved, which means that all
programs expecting a library's schema need not be changed to work with the
application's schema. This is true for the library itself as well as possible
other libraries and applications consuming its data.

However, this requires one additional abstraction. The library (and possible
other consuming programs) must be able to handle both cases:

1. "Library only" mode, where the library's schema is the only (and therefore
   the top-level) schema inside a storage.
2. "Application" mode, where the library's schema is only a subset of a broader
   schema.

The difference is that in "library only" mode, the library has to create (or
load) and possibly save the entire storage, while in "application" mode, this is
the applications responsibility. To simplify this, dsch provides the
:class:`~dsch.frontend.PseudoStorage` class which automates the process. It can
be initialized with either a string argument::

   pseudo = dsch.PseudoStorage('example_lib_data.h5', schema_lib1)

... or with a data node argument::

   storage = dsch.load('example_app_data.h5')
   pseudo = dsch.PseudoStorage(storage.data.lib1, schema_lib1)

In both cases, ``pseudo.data`` now provides the data node corresponding to the
top-level node in ``schema_lib1``. The library does not need to perform any
further checks or decisions in this regard.

Additionally, :class:`~dsch.frontend.PseudoStorage` can be used as a context
manager::

   with pseudo as p:
       p.data.spam = 2342
       p.data.eggs = True

This automatically handles saving the storage when leaving the context, if
appropriate (i.e. if we are in "library only" mode).

Using :class:`~dsch.frontend.PseudoStorage` is the recommended way for using
dsch in library code.

Multiple schema versions
------------------------

Sometimes, schema changes cannot be avoided, so a new version must be designed.
However, backwards compatibility is usually desired, at least on the data
consumption side.

When using :func:`~dsch.frontend.load`, this can be achieved by simply not
setting the ``required_schema`` argument. Then, the storage's
:attr:`~dsch.storage.Storage.schema_node` attribute can be checked for
compatibility and possible adaption of the subsequent data handling steps.

When using :class:`~dsch.frontend.PseudoStorage`, a different approach is
required since the ``schema_node`` attribute cannot be omitted upon object
creation. This is because the :class:`~dsch.frontend.PseudoStorage` must "know"
the desired schema for cases in which it has to create a new storage.

To use multiple schema versions with :class:`~dsch.frontend.PseudoStorage`,
supply the ``schema_alternatives`` attribute on object creation::

   current_schema = dsch.schema.Compilation(...)
   old_schema = dsch.schema.Compilation(...)
   pseudo = dsch.PseudoStorage(storage_path, schema_node=current_schema,
                               schema_alternatives=(old_schema,))

Now, when loading from a storage or a data node, ``pseudo`` will first check the
detected schema against ``current_schema`` (because that was specified as
``schema_node``). If these do not match, every schema in ``schema_alternatives``
is tried, and only if none of these match, an
:exc:`~dsch.exceptions.InvalidSchemaError` is raised. For *creating* new
storages, only the ``schema_node`` is used and ``schema_alternatives`` are not
considered.

An arbitrary number of alternative schemas can be specified through
``schema_alternatives``, and each can be given as either the schema node object
or as a string, representing the corresponding schema node's hash.


Querying field state
====================

"Complete" and "empty" fields
-----------------------------

As presented in the tutorial, all data nodes have an ``empty`` attribute that,
if ``True``, indicates the absence of a value for this node. For
:class:`~dsch.data.Compilation`, ``empty`` works recursively.

.. note::
   To restore a non-``empty`` node back to the ``empty`` state, i.e. entirely
   remove the stored data, use the ``clear()`` method.

For practical use, it can be helpful to know whether a dataset contains all
required information, i.e. whether it is ``complete``. Therefore, data nodes
also have a ``complete`` attribute, which indicates the presence of a value.

Note that the value of ``complete`` is only the inverse of ``empty`` for regular
data nodes. For :class:`~dsch.data.Compilation`, they are evaluated recursively
for all sub-nodes, which means that ``complete`` is ``True`` if *all* sub-nodes
are complete, while ``empty`` is ``True`` if *all* sub-nodes are empty. Thus,
both can be ``False`` at the same time.

Optional fields in Compilations
-------------------------------

Some schemas may contain optional fields, i.e. fields that are not required for
a dataset to be considered "complete". For example, a measurement result might
contain a "comment" field that is not strictly required for the dataset to make
sense. In this case, :attr:`~dsch.data.Compilation.complete` should return
``True`` even if no comment is provided.

This behaviour can be achived by simply passing a list of ``optionals`` during
schema node initialization of the Compilation::

   schema = dsch.schema.Compilation({
       'time': dsch.schema.Array(dtype='float', unit='s'),
       'voltage': dsch.schema.Array(dtype='float', unit='V'),
       'comment': dsch.schema.String(),
   }, optionals=['comment'])


Validation
==========

Ensuring a compatible schema
----------------------------

When loading a storage, dsch can ensure that it conforms to a specific schema.
Then, consuming code can rely on the data to really be structured in the
expected way.  Schemas are automatically identified by a SHA256 hash, which can
be queried via any schema node's :meth:`~dsch.schema.SchemaNode.hash` or a
storage's :meth:`~dsch.storage.Storage.schema_hash`.  Once determined, it can be
given to :func:`~dsch.frontend.load` as the ``require_schema`` argument, causing
dsch to raise a :exc:`RuntimeError` if the to-be-loaded storage has a different
schema::

   hash = known_good_storage.schema_hash()
   unknown_storage = dsch.load(path_to_storage, require_schema=hash)


Inter-node validation
---------------------

Usually, validation only covers a single node at a time, so each node's value is
validated against the exact node's constraints.  This is insufficient for e.g.
digital signals, like a measured voltage over time, which could be represented
as two :class:`~dsch.schema.Array` instances ``voltage`` and ``time``.  In this
case, ``time`` is the independent variable and ``voltage`` depends on ``time``,
implicitly requiring the length of the arrays to be equal and the dimensionality
of ``time`` to be 1.

Automatic validation of these constraints can be achieved by providing a
``depends_on`` argument to the dependent variable's schema node::

   schema = dsch.schema.Compilation({
       'time': dsch.schema.Array(dtype='float'),
       'voltage': dsch.schema.Array(dtype='float', depends_on=('time',))
   })

That argument must be an iterable of field names corresponding to all
independent variables, so this also works for arrays of higher dimensionality.
For example, a 2-dimensional matrix could have two entries in ``depends_on``,
one for each dimension.  If no independent variable exists for a particular
dimension, ``None`` may be specified instead of a field name.
