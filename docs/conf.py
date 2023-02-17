# Configuration file for the Sphinx documentation builder.
# Partially based on https://github.com/icaros-usc/pyribs/blob/master/docs/conf.py
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import sphinx_material

from importlib import metadata

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'assembled'
copyright = '2023, Lennart Purucker'
author = 'Lennart Purucker'
version = metadata.version(project)
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_material",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_toolbox.more_autodoc.autonamedtuple",
    "sphinx_autodoc_typehints",
    "sphinx_codeautolink",
    "myst_parser"
]

# Napoleon
napoleon_use_ivar = True

# Auto-generate heading anchors.
myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_show_sourcelink = True
html_sidebars = {"**": ["globaltoc.html", "localtoc.html", "searchbox.html"]}

html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = "sphinx_material"
html_logo = "_static/imgs/logo_transparent.png"
html_favicon = "_static/imgs/favicon.ico"
html_title = f"assembled"


# material theme options (see theme.conf for more information)
html_theme_options = {
    "nav_title": "assembled",
    "base_url": "https://isg-siegen.github.io",
    "repo_url": "https://github.com/ISG-Siegen/assembled",
    "repo_name": "assembled",
    "google_analytics_account": None,
    "repo_type": "github",
    "globaltoc_depth": 2,
    "color_primary": "deep-orange",
    "color_accent": "orange",
    "touch_icon": None,
    "master_doc": False,
    "nav_links": [{
        "href": "index",
        "internal": True,
        "title": "Home"
    },],
    "heroes": {
        "index":
            "A framework to find better ensembles for (Automated) Machine Learning."
    },
    "version_dropdown": False,
    "version_json": None,
    "table_classes": ["plain"],
}

html_last_updated_fmt = None

html_use_index = True
html_domain_indices = True

html_static_path = ['_static']


# -- Extension config -------------------------------------------------

# Autodoc and autosummary
autodoc_member_order = "groupwise"  # Can be overridden by :member-order:
autodoc_default_options = {
    "inherited-members": True,
}
autosummary_generate = True
