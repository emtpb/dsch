****
dsch
****

DSCH provides structured, metadata-enhanced data storage for Python 3.


Features
========

* Application-specific, user-defined data schemas

   * Data fields (e.g. NumPy arrays, date/time, strings)
   * Data validation (e.g. array size limits)
   * Container elements (dict- or list-like)
   * Arbitrary nested data structures
   * Ensures consistent data input for later processing

* Metadata included in schemas

   * Consistent metadata for all datasets/files
   * Implicit documentation of dataset contents
   * Use e.g. for the unit of a physical quantity

* Support for multiple storage backends

   * `HDF5 files <https://hdfgroup.org>`_
   * `NumPy npz files <https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html>`_
   * `MATLAB mat files <https://www.mathworks.com/products/matlab.html>`_
   * Modular interface for adding new backends

* Unified user frontend

   * Data handling is independent from the backend in use
