"""Build the documentation without importing numerical modules."""
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "api"))
from check_coverage import check_reference

metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
project = "DynCFS"
author = "Jiangcheng Zhou"
copyright = "2026, Jiangcheng Zhou"
release = metadata["version"]
version = release
language = "en"
root_doc = "index"
extensions = ["myst_parser", "sphinx.ext.mathjax", "sphinx.ext.githubpages", "sphinx_design"]
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "README.md", "requirements.txt"]
html_theme = "pydata_sphinx_theme"
html_title = f"DynCFS {release}"
html_short_title = "DynCFS"
html_baseurl = "https://zhou-jiangcheng.github.io/dyncfs/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/Zhou-Jiangcheng/dyncfs",
    "show_nav_level": 1,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "collapse_navigation": True,
    "announcement": f"Documentation for the main branch · package {release}",
}
html_context = {
    "github_user": "Zhou-Jiangcheng", "github_repo": "dyncfs",
    "github_version": "main", "doc_path": "docs",
}
html_show_sourcelink = True
html_last_updated_fmt = None

def setup(app):
    app.connect("builder-inited", lambda app: check_reference(ROOT))
