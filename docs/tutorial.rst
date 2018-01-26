.. _tutorial:

********
Tutorial
********

This tutorial will give you a quick overview on working with DSCH.
Just make sure you have :ref:`installed <installation>` it and get started!


Defining a Schema
=================

Let's start with a simple example.
First, we must import :mod:`dsch` itself::

   import dsch

Suppose we want to store data of a little weather station that measures
temperature and humidity, plus the date and time of each measurement.
With this set of quantities in mind, we can construct a `schema` that
essentially defines the structure of our dataset. In this example, the schema
might look like this::

   schema = dsch.schema.Compilation({
       'time': dsch.schema.DateTime(),
       'temperature': dsch.schema.Scalar(dtype='float', unit='°C'),
       'humidity': dsch.schema.Scalar(dtype='float', unit='%', min_value=0,
                                      max_value=100)
   })

Here, we use a :class:`~dsch.schema.Compilation` as a container for our data
fields, allowing us to name the individual fields. Also, we define the units to
be used for the physical quantities and sensible value limits for the humidity.
For numerical data, the ``dtype`` (corresponding to NumPy :class:`~numpy.dtype`)
must also be given.


Opening a Storage
=================

Dsch can store data using a number of backends, e.g. different file formats.
However, since backends could also be implemented for databases or other
non-file storage engines, we avoid the term "file" and use "storage" instead.
A storage always holds data corresponding to exactly one schema.

To open a storage, we can either use :func:`dsch.load() <dsch.frontend.load>` or
:func:`dsch.create() <dsch.frontend.create>`, depending on whether the storage
already exists. In both cases, we must provide a location for the storage, e.g.
the path to a file. Since we started from scratch, we must use
:func:`dsch.create() <dsch.frontend.create>`, which requires our previously
defined ``schema`` as an argument::

   storage = dsch.create('test.h5', schema)

If we wanted to open an existing storage, we would not need to provide the
schema, as it is automatically loaded::

   storage = dsch.load('test.h5')

Note that in both cases, we did not have to explicitly state the backend to be
used. This is because dsch automatically detects that this is a file path ending
in ".h5" and chooses the HDF5 backend accordingly. For instance, if we wrote
``test.npz``, the NumPy npz backend would be chosen instead. This auto-detection
can also be overridden if required, see :func:`~dsch.frontend.create` and
:func:`~dsch.frontend.load`.


Accessing Data
==============

Once we have a ``storage`` object, we can start accessing the data. All data is
provided via the ``data`` attribute of a storage, which is structured in the
exact way that we previously defined in our schema. So, in our example,
``storage.data`` (the top-level data node) is a Compilation containing three
child nodes (``time``, ``temperature`` and ``humidity``).

Child nodes of Compilations are represented as attributes, so they can be easily
accessed with the "dotted" notation.
Note, however, that these `data nodes` are not simply the stored values, but
objects with additional functionality. The actual stored value is available
through the data node's ``value`` attribute::

   >>> storage.data.time.value
   [...]
   NodeEmptyError: Node is empty. The value of empty nodes is undefined.

Well, we should have expected this! Since we just created a new, empty file,
there is no data present. In fact, we can check whether a node is empty::

   >>> storage.data.time.empty
   True

This even works for Compilation nodes, where all child nodes are checked
recursively::

   >>> storage.data.empty
   True

The ``empty`` attribute is an example of functionality that data nodes provide
beyond simply storing a value.
Depending on the node type and the backend in use, there are different
functional ranges.

Of course, we can also assign new variables for any node, providing a shortcut
for access::

   >>> temp = storage.data.temperature
   >>> temp.empty
   True


Modifying Data
==============

The data stored in a data node can be changed by setting the ``value``
attribute. This is also the way to apply an initial value to an empty node::

   import datetime
   storage.data.time.value = datetime.datetime.now()
   storage.data.temperature.value = 21
   storage.data.humidity.value = 42

Now, we can inspect the filled data structure::

   >>> storage.data.empty
   False

   >>> storage.data.temperature.value
   21.0

An alternative to setting all values individually is to use the Compilation's
``replace`` method, which accepts a :class:`dict`::

   storage.data.replace({
       'time': datetime.datetime.now(),
       'temperature': 21,
       'humidity': 42
   })

This is equivalent to the example above.


Data Validation
===============

All data can be validated against the constraints defined in the schema.
For example, our schema states that the value for ``humidity`` must be in the
range from 0 to 100.
Since we previously set that value to 42, validation succeeds (i.e. terminates
silently)::

   >>> storage.data.humidity.validate()

However, if we set an out-of range value, a
:class:`~dsch.schema.ValidationError` is raised::

   >>> storage.data.humidity.value = 123
   >>> storage.data.humidity.validate()
   [...]
   ValidationError: Maximum value exceeded. (Expected: 100. Got: 123.0)

Of course, we can also validate the entire storage in a single step::

   >>> storage.validate()
   [...]
   SubnodeValidationError: Field "humidity" failed validation: Maximum value exceeded. (Expected: 100. Got: 123.0)

Note that now, a :class:`~dsch.data.SubnodeValidationError` is raised, providing
details on the affected node.


Storing Data
============

For all current backends, changes to the data inside a storage are not
automatically written to disk.
To do that, you must call :meth:`~dsch.storage.FileStorage.save` explicitly::

   >>> storage.save()
   [...]
   SubnodeValidationError: Field "humidity" failed validation: Maximum value exceeded.

Oh, right, we still have that invalid value set for ``humidity``! As we can see,
data is, by default, automatically validated before saving. This prevents us
from accidentally producing files with invalid for physically impossible values.
Of course, when we provide a sensible value again, we can easily save our file::

   >>> storage.data.humidity.value = 42
   >>> storage.save()


Conclusion
==========

Handling data with dsch is easy! Just define a schema, open a storage for it,
and fill it with data - that's it for basic usage patterns!

Of course, there are a few more features in dsch that you might want to use.
These are presented in short blocks in :ref:`Advanced Topics <advanced>`.
