"""
docs_links.py — Website links of README and CHANGELOG as links within the website build.

README.md and CHANGELOG.md are also read on GitHub, so they link into the documentation with
the absolute URLs of the website (``SITE``). The website includes parts of them
(``pymdownx.snippets``, see ``mkdocs.yml``). This Markdown extension turns those URLs into
relative links to the pages of the same build: a local build (``zensical serve``) and every
published version link to their own pages, also before a page is online, and the link check
of the build covers them. Registered in ``mkdocs.yml``; it runs after the snippets are
included. Without the page context of Zensical it leaves the text as it is.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import List, Optional

from markdown import Extension
from markdown.preprocessors import Preprocessor

SITE = "https://stofe94.github.io/crash-def-net/"
DOCS = Path(__file__).resolve().parents[1] / "docs"
# A website URL as the target of a Markdown link: the path of the page, then an anchor.
LINK = re.compile(r"\]\(" + re.escape(SITE) + r"([^)#\s]*)(#[^)\s]*)?\)")
SNIPPET_PRIORITY = 32   # pymdownx.snippets; preprocessors with a lower priority run later


def source_page(url_path: str, docs: Path = DOCS) -> Optional[str]:
    """The page below ``docs`` of a website path (``concepts/overview/`` ->
    ``concepts/overview.md``, ``about/`` -> ``about/index.md``), or None."""
    path = url_path.strip("/")
    candidates = (f"{path}.md", f"{path}/index.md") if path else ("index.md",)
    return next((page for page in candidates if (docs / page).is_file()), None)


def local_links(text: str, page: str, docs: Path = DOCS) -> str:
    """``text`` of the page ``page`` (path below ``docs``) with every website link that
    belongs to a page of ``docs`` relative; other links stay as they are."""
    here = posixpath.dirname(page.replace("\\", "/")) or "."

    def relative(match: re.Match) -> str:
        target = source_page(match.group(1), docs)
        if target is None:
            return match.group(0)
        return f"]({posixpath.relpath(target, here)}{match.group(2) or ''})"

    return LINK.sub(relative, text)


class SiteLinks(Preprocessor):
    def run(self, lines: List[str]) -> List[str]:
        try:
            from zensical.extensions.context import ContextPreprocessor
        except ImportError:
            return lines
        context = ContextPreprocessor.from_markdown(self.md)
        if context is None:
            return lines
        return [local_links(line, context.page.path) for line in lines]


class SiteLinksExtension(Extension):
    def extendMarkdown(self, md) -> None:
        md.preprocessors.register(SiteLinks(md), "site_links", SNIPPET_PRIORITY - 1)


def makeExtension(**kwargs) -> SiteLinksExtension:
    return SiteLinksExtension(**kwargs)
