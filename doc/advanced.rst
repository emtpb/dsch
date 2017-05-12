.. _advanced:

Advanced Topics
===============

Nested schemas
--------------

The example schema from the :ref:`Tutorial <tutorial>` is a very simple one, so let's extend it!
Suppose we do not only want to store a single measurement result of our weather station, but multiple results taken at different times.
A simple solution for this would be to wrap the previous schema in a :class:`dsch.schema.List`::

    schema_list = dsch.schema.List(
        dsch.schema.Compilation({
            'time': dsch.schema.DateTime(),
            'temperature': dsch.schema.Scalar(dtype='float', unit='°C'),
            'humidity': dsch.schema.Scalar(dtype='float', unit='%', min_value=0,
                                           max_value=100)
        })
    )

If we now create a storage for this schema, we can use :meth:`dsch.data.List.append` to add individual measurement results to the list::

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
    storage_list.data[0].time.replace(datetime.datetime.now())
    storage_list.data[0].temperature.replace(21)
    storage_list.data[0].humidity.replace(42)

By nesting Lists and Compilations, arbitrary schemas can be composed.
