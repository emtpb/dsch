*********
Changelog
*********

This project follows the guidelines of `Keep a changelog`_ and adheres to
`Semantic versioning`_.

.. _Keep a changelog: http://keepachangelog.com/
.. _Semantic versioning: https://semver.org/


`Unreleased`_
=============

Added
-----
* New node type for `bytes` data.
* In-memory backend, for handling data without needing e.g. a file on disk.
* Support for copying data between different storages.
* Support for creating new storages from existing ones, aka. "save as".

Changed
-------
* Data nodes in Compilations and Lists can no longer be overwritten
  accidentally when trying to overwrite their stored value.
* Improve structure and conciseness of docs.


`0.1.3`_ - 2018-01-11
=====================

Changed
-------
* Attempting to open a non-existent file now shows a sensible error message.
* Attempting to create an existing file now shows a sensible error message.

Fixed
-----
* Fix error when handling partially filled compilations.
* Fix typo in documentation.


`0.1.2`_ - 2017-08-25
=====================

Fixed
-----
* Fix incorrect ordering of list items.


`0.1.1`_ - 2017-06-09
=====================

Added
-----
* Cover additional topics in documentation.

Fixed
-----
* Fix error when handling single-element lists with `mat` backend.


`0.1.0`_ - 2017-05-18
=====================

Added
-----
* First preview release.


.. _Unreleased: https://github.com/emtpb/dsch
.. _0.1.3: https://github.com/emtpb/dsch/releases/tag/0.1.3
.. _0.1.2: https://github.com/emtpb/dsch/releases/tag/0.1.2
.. _0.1.1: https://github.com/emtpb/dsch/releases/tag/0.1.1
.. _0.1.0: https://github.com/emtpb/dsch/releases/tag/0.1.0
