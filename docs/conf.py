project = "Topic Boundaries"
extensions = ["myst_parser", "sphinx_design", "sphinx.ext.autodoc"]
root_doc = "index"
exclude_patterns = ["_build"]

html_title = project
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "show_toc_level": 2,
    "primary_sidebar_end": [],
    "secondary_sidebar_items": ["page-toc"],
}
html_show_sourcelink = False
