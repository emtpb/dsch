.. _advanced:

Advanced Topics
===============

Nested schemas
--------------

The example schema from the :ref:`Tutorial <tutorial>` is a very simple one, so let's extend it!
Suppose we do not only want to store a single measurement result of our weather station, but multiple results taken at different times.
A simple solution for this would be to wrap the previous schema in a :class:`~dsch.schema.List`::

    schema_list = dsch.schema.List(
        dsch.schema.Compilation({
            'time': dsch.schema.DateTime(),
            'temperature': dsch.schema.Scalar(dtype='float', unit='°C'),
            'humidity': dsch.schema.Scalar(dtype='float', unit='%', min_value=0,
                                           max_value=100)
        })
    )

If we now create a storage for this schema, we can use :meth:`~dsch.data.List.append` to add individual measurement results to the list::

    storage_list = dsch.create('list.h5', schema_list)
    storage_list.data.append({
        'time': datetime.datetime.now(),
        'temperature': 21,
        'humidity': 42
    })

Each list item is a Compilation that behaves exactly like in the previous sections::

    >>> storage_list.data[0].temperature.value
    21.0

Alternatively, if you prefer to work with the data nodes directly, the argument to ``append`` can be omitted::

    storage_list.data.append()
    storage_list.data[0].time.value = datetime.datetime.now()
    storage_list.data[0].temperature.value = 21
    storage_list.data[0].humidity.value = 42

By nesting Lists and Compilations, arbitrary schemas can be composed.


Optional fields in Compilations
-------------------------------

Some data structures may contain truly optional fields.
For example, a measurement result might contain a "comment" field that is not strictly required fir the dataset to make sense.
In this case, :attr:`~dsch.data.Compilation.complete` should return ``True`` even if no comment is provided, because the measurement result is, in fact, complete without it.
This behaviour can be achived by simply passing a list of ``optionals`` during schema node initialization of the Compilation::

    subnodes = {
        'time': dsch.schema.Array(dtype='float', unit='s'),
        'voltage': dsch.schema.Array(dtype='float', unit='V'),
        'comment': dsch.schema.String(),
    }
    schema = dsch.schema.Compilation(subnodes, optionals=['comment'])

In this example, the ``comment`` field would be ignored when checking :attr:`~dsch.data.Compilation.complete`.
Each entry of ``optionals`` must match the name of one of the Compilation's subnodes.


Checking for specific schemas
-----------------------------

When loading a storage, DSCH can ensure it conforms to a specific schema.
Then, subsequent processing code can rely on the data to really be structured in the expected way.
Schemas are automatically identified by a SHA256 hash, which can be queried by calling :meth:`~dsch.storage.Storage.schema_hash`.
Once determined, it can be given to :func:`~dsch.frontend.load` as the ``require_schema`` argument, causing DSCH to raise a :exc:`RuntimeError` if the to-be-loaded storage has a different schema::

    hash = known_good_storage.schema_hash()
    unknown_storage = dsch.load(path_to_storage, require_schema=hash)
