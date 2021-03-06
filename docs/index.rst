********************************
Welcome to dsch's documentation!
********************************

DSCH (data schema consistency helper) provides structured, metadata-enhanced data storage in different formats.
Users define an application-specific data schema, combining different fields (e.g. NumPy arrays, strings, dates) into the required structure.
Constraints may be applied to individual fields, e.g. limiting the size of an array.
Also, the schema can include metadata, e.g. units of physical quantities, to produce consistent and self-documenting datasets.
Data can be stored in different formats (HDF5, NumPy or MATLAB files) through a single, unified interface.


Contents
========

.. toctree::
   :maxdepth: 2

   readme
   installation
   tutorial
   advanced
   api
   authors
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
