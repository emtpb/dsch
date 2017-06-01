.. _installation:

Installation
============

From PyPI
---------

DSCH wheels are available on `PyPI <https://pypi.python.org/pypi/dsch>`_, so it can be installed on the command line via `pip <https://docs.python.org/3/installing/>`_::

    $ pip install dsch

On some systems where Python 2 is still the default, you may need to use ``pip3`` instead::

    $ pip3 install dsch

Note that DSCH is neither developed for nor tested against any versions of Python 2.


From Source
-----------

If you wish to install from source instead of using a wheel, you can either use the source distribution from PyPI::

    $ pip install --no-binary=dsch dsch

or you can clone or download the source from the `GitHub repository <https://github.com/emtpb/dsch>`_ and install it using ``pip`` (or ``pip3``)::

    $ pip install .

or::

    $ python setup.py install
