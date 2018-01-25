****
dsch
****


Introduction
============

Dsch provides a way to store data and its metadata in a structured, reliable
way. It is built upon well-known data storage engines, such as the `HDF5`_ file
format, providing performance and long-term stability.

The core feature is the schema-based approach to data storage, which means that
a pre-defined schema specification is used to determine:

* which data fields are available
* the (hierarchical) structure of data fields
* metadata of the stored values (e.g. physical units)
* expected data types and constraints for the stored values

In fact, this is similar to an API specification, but it can be attached to and
stored with the data. Programs *writing* datasets benefit from data validation
and the high-level interface. *Reading* programs can determine the given data's
schema upfront, and process accordingly. This is especially useful with schemas
evolving over time.

For persistent storage, dsch supports multiple storage engines via its
`backends`, but all through a single, transparent interface. Usually, there are
no client code changes required to support a new backend, and custom backends
can easily be added to dsch.
Currently, backends exist for these storage engines:

* `HDF5`_ files (through `h5py`_)
* `NumPy .npz`_ files
* `MATLAB .mat`_ files (through `SciPy`_)

Note that dsch is only a thin layer, so that users can still benefit from the
performance of the underlying storage engine. Also, files created with dsch can
always be opened directly (i.e. without dsch) and still provide all relevant
information, even the metadata!

.. _HDF5: https://hdfgroup.org
.. _h5py: http://www.h5py.org
.. _NumPy .npz: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
.. _MATLAB .mat: https://www.mathworks.com/products/matlab.html
.. _SciPy: https://docs.scipy.org/doc/scipy-0.19.0/reference/io.html


Reasoning
=========

Dsch is a response to the challenges in low-level data acquisition scenarios,
which are commonly found in labs at universities or R&D departments. Frequent
changes in both hardware and software are commonplace in these environments, and
since those changes are often made by different people, the data acquisition
hardware, software and data consumption software tend to get out of sync. At the
same time, datasets are often stored (and used!) for many years, which makes
backwards-compatibility a significant issue.

Dsch aims to counteract these problems by making the data exchange process more
explicit. Using pre-defined schemas ensures backward-compatibility as long as
possible, and when it can no longer be retained, provides a clear way to detect
(and properly handle) multiple schema versions. Also, schema based validation
allows to detect possible errors upfront, so that most non-security-related
checks do not have to be re-implemented in data consuming applications.

Note that dsch is targeted primarily at these low-level applications. When using
high-level data processing or even data science and machine learning techniques,
data is often pre-processed and aggregated with regard to a specific
application, which often eliminates the need for some of dsch's features, such
as the metadata storage. One might think of dsch as the tool to handle data
*before* it is filled into something like `pandas`_.

.. _pandas: https://pandas.pydata.org/
