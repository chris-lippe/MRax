import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mrax import __version__ as release

project = "MRax"
author = "Chris Lippe"
contact_email = "chris.lippe@uni-muenster.com"
license_name = "MIT"
version = ".".join(release.split(".")[:2])
copyright = ""
docs_base_url = os.environ.get("MRAX_DOCS_BASE_URL", "").rstrip("/")

extensions = [
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_member_order = "bysource"
nbsphinx_execute = "never"  # render notebooks without executing them during builds
# Keep notebook highlighting portable across supported Pygments releases.  nbsphinx
# reads the notebook metadata first, but this remains the fallback for new notebooks.
nbsphinx_codecell_lexer = "python"
nbsphinx_prolog = r"""
{% set nbtitle = env.docname.split('/')[-1].replace('_', ' ').title() %}
{{ nbtitle }}
{{ '=' * nbtitle|length }}
"""

# Pandoc may be unavailable in minimal environments; fall back to HTML for Markdown cells.
try:
    import nbsphinx as _nbsphinx
    from nbconvert.filters.markdown import markdown2html_mistune

    _orig_convert = _nbsphinx.convert_pandoc

    import re

    _CODE_FENCE_RE = re.compile(r"(^```.*?$.*?^```|^~~~.*?$.*?^~~~)", re.MULTILINE | re.DOTALL)
    _INLINE_CODE_RE = re.compile(r"`[^`]+`")
    _MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
    _MATH_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
    _MATH_INLINE_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL)
    _MATH_PAREN_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)

    def _escape_math_html(text: str) -> str:
        return text.replace("<", "&lt;").replace(">", "&gt;")

    def _extract_math(segment: str, math_entries: list[tuple[str, str]]) -> str:
        code_entries: list[str] = []

        def _code_repl(match: re.Match) -> str:
            idx = len(code_entries)
            code_entries.append(match.group(0))
            return f"@@CODE_{idx}@@"

        segment = _INLINE_CODE_RE.sub(_code_repl, segment)

        def _block_repl(match: re.Match) -> str:
            idx = len(math_entries)
            math_entries.append(("block", match.group(1)))
            return f"@@MATH_{idx}@@"

        def _inline_repl(match: re.Match) -> str:
            idx = len(math_entries)
            math_entries.append(("inline", match.group(1)))
            return f"@@MATH_{idx}@@"

        segment = _MATH_BLOCK_RE.sub(_block_repl, segment)
        segment = _MATH_BRACKET_RE.sub(_block_repl, segment)
        segment = _MATH_INLINE_RE.sub(_inline_repl, segment)
        segment = _MATH_PAREN_RE.sub(_inline_repl, segment)

        for idx, code in enumerate(code_entries):
            segment = segment.replace(f"@@CODE_{idx}@@", code)
        return segment

    def _preserve_math(text: str) -> tuple[str, list[tuple[str, str]]]:
        math_entries: list[tuple[str, str]] = []
        parts: list[str] = []
        last = 0
        for match in _CODE_FENCE_RE.finditer(text):
            parts.append(_extract_math(text[last:match.start()], math_entries))
            parts.append(match.group(0))
            last = match.end()
        parts.append(_extract_math(text[last:], math_entries))
        return "".join(parts), math_entries

    def _restore_math(html: str, math_entries: list[tuple[str, str]]) -> str:
        for idx, (kind, content) in enumerate(math_entries):
            safe = _escape_math_html(content)
            if kind == "block":
                repl = f'<div class="math notranslate nohighlight">\\[{safe}\\]</div>'
                html = html.replace(f"<p>@@MATH_{idx}@@</p>", repl)
            else:
                repl = f'<span class="math notranslate nohighlight">\\({safe}\\)</span>'
            html = html.replace(f"@@MATH_{idx}@@", repl)
        return html

    def _md_to_rst_via_html(text, from_format, to_format):
        if str(from_format).startswith(("md", "markdown")) and str(to_format) == "rst":
            processed, math_entries = _preserve_math(text)
            html = markdown2html_mistune(processed)
            html = _restore_math(html, math_entries)
            indented = "\n".join(f"   {line}" for line in html.splitlines())
            return ".. raw:: html\n\n" + indented + ("\n" if not indented.endswith("\n") else "")
        try:
            return _orig_convert(text, from_format, to_format)
        except Exception:
            return text

    _nbsphinx.convert_pandoc = _md_to_rst_via_html
except Exception:
    pass

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/MRax_logo_nav.png"
html_show_copyright = False
html_context = {
    "contact_email": contact_email,
    "license_name": license_name,
}
html_theme_options = {
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "logo": {
        "image_light": "_static/MRax_logo_nav.png",
        "image_dark": "_static/MRax_logo_nav.png",
        "text": "",
    },
    "secondary_sidebar_items": [],
    "primary_sidebar_end": ["indices"],
    "footer_start": ["license-contact"],
    "icon_links": [],
}

html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
    "api": ["page-toc.html"],
    "models": ["page-toc.html"],
    "overview": ["page-toc.html"],
}

html_css_files = ["custom.css"]
