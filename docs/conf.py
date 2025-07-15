# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib
import inspect
import os

project = "Synax"
copyright = "2025, Carlos Martin"
author = "Carlos Martin"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.linkcode",
    "myst_parser",
]
autosummary_generate = True
autoapi_dirs = ["../src/synax", "."]
autoapi_options = [
    # "members",
    # "undoc-members",
    # "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_own_page_level = "method"

linkcode_url = "https://github.com/carlosgmartin/synax/blob/master/src/synax/{path}#L{start}-L{end}"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    assert info["module"] == ""
    names = info["fullname"].split(".")
    obj = importlib.import_module(names[0])
    for name in names[1:]:
        try:
            obj = getattr(obj, name)
        except AttributeError:
            return None

    obj_path = inspect.getsourcefile(obj)
    assert obj_path is not None

    mod = inspect.getmodule(obj)
    assert mod is not None
    mod = importlib.import_module(mod.__name__.split(".")[0])
    mod_path = inspect.getsourcefile(mod)
    assert mod_path is not None
    pkg_path = os.path.dirname(mod_path)

    rel_path = os.path.relpath(obj_path, pkg_path)

    source, start = inspect.getsourcelines(obj)
    end = start + len(source) - 1

    url = linkcode_url.format(path=rel_path, start=start, end=end)

    return url


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/carlosgmartin/synax",
            "html": open("github-icon.svg").read(),
            "class": "",
        },
    ],
}
html_show_sphinx = False
html_show_sourcelink = False

html_logo = "logo.svg"
html_favicon = "logo.svg"
