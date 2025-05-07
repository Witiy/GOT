import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
import pygot  # isort:skip # noqa E402

from urllib.request import urlretrieve



# The suffix of source filenames.
source_suffix = ['.rst',".md"]
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'GOT'
copyright = '2025, Ruihong Xu'
author = 'Ruihong Xu'
release = '0.2.4'
html_favicon = 'favicon.png'
bibtex_bibfiles = ['ref.bib']
    
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'recommonmark',
    'sphinx_markdown_tables',
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
