from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst')) as readme_file:
    long_description = readme_file.read()

setup(
    name='dsch',

    description='Structured, metadata-enhanced data storage.',
    long_description=long_description,

    url='http://emt.uni-paderborn.de',
    author='Manuel Webersen',
    author_email='webersen@emt.uni-paderborn.de',
    license='BSD',

    # Automatically generate version number from git tags
    use_scm_version=True,

    packages=[
        'dsch'
    ],

    # Runtime dependencies
    install_requires=[
        'h5py',
        'numpy',
        'scipy',
    ],

    # Setup/build dependencies
    # setuptools_scm is required for git-based versioning
    setup_requires=['setuptools_scm'],

    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
)
