import datetime
import os
import sys
import setuptools_scm

# Add source code directory to path (required for autodoc)
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.imgmath',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

# Show members of modules/classes and parent classes by default
autodoc_default_flags = ['members']

# Set up napoleon for parsing Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Configure remote documenation via intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'h5py': ('http://docs.h5py.org/en/latest/', None),
}

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = None
today_fmt = '%Y-%m-%d'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False

# -- Project-specific configuration

project = 'DSCH'
description = 'Structured, metadata-enhanced data storage.'
author = 'Manuel Webersen'
copyright = '{year} {author}'.format(year=datetime.date.today().year,
                                     author=author)
project_without_spaces = ''.join(c for c in project if c.isalnum())

# Get version number from git via setuptools_scm
# Do not differentiate between shortened version and full release numbers
version = setuptools_scm.get_version(root='..', relative_to=__file__)
release = version

# -- Options for HTML output

html_theme = 'alabaster'
html_theme_options = {
    'description': description
}
html_sidebars = {
    '**': ['about.html', 'navigation.html', 'searchbox.html']
}
htmlhelp_basename = '{0}doc'.format(project)

# -- Options for LaTeX output

latex_elements = {
     'papersize': 'a4paper',
}
latex_documents = [
    (master_doc, '{0}.tex'.format(project_without_spaces),
     '{0} Documentation'.format(project), author, 'manual'),
]
