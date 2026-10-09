from __future__ import annotations

from pygments.lexers.special import TextLexer
from sphinx.highlighting import lexers

import kimech

# Keep GitHub-native Mermaid code fences valid in Sphinx without requiring
# a JavaScript renderer. Sphinx shows the editable diagram source as text.
lexers["mermaid"] = TextLexer()

project = "Kimech"
author = "Pedro Jorge De Los Santos"
release = kimech.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "furo"
html_title = f"Kimech {release}"
html_theme_options = {
    "source_repository": "https://github.com/JorgeDeLosSantos/kimech/",
    "source_branch": "main",
    "source_directory": "docs/user/",
}

exclude_patterns = ["_build"]
