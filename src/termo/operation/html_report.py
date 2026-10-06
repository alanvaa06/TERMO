"""A small Markdown subset as HTML, and the UTF-8 writer of the report.

Used for the analyst's comment in the visual report. The subset: `#`/`##` headings,
`**bold**`, pipe tables with a `|---|` separator row, `- ` bullets and blank-line
separated paragraphs; any other line is a paragraph. Text is escaped, so a `<` in a
cell stays a `<`; accents and the `→` are kept.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from pathlib import Path

BOLD = re.compile(r"\*\*(.+?)\*\*")
TABLE_SEPARATOR = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")


def _inline(text: str) -> str:
    """Escaped text with `**bold**` turned into <strong>."""
    return BOLD.sub(r"<strong>\1</strong>", html.escape(text, quote=False))


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _row(line: str, tag: str) -> str:
    return "<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in _cells(line)) + "</tr>"


def _table(lines: Sequence[str]) -> str:
    """A pipe table: a header row when the second line is the separator, then the body."""
    body = list(lines)
    head = ""
    if len(body) >= 2 and TABLE_SEPARATOR.match(body[1].strip()):
        head = f"<thead>{_row(body[0], 'th')}</thead>"
        body = body[2:]
    rows = "".join(_row(line, "td") for line in body)
    return f"<table>{head}<tbody>{rows}</tbody></table>"


def md_to_html(text: str) -> str:
    """The Markdown subset as HTML blocks, one per line of output."""
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
        elif stripped.startswith("# "):
            blocks.append(f"<h1>{_inline(stripped[2:].strip())}</h1>")
            i += 1
        elif stripped.startswith("## "):
            blocks.append(f"<h2>{_inline(stripped[3:].strip())}</h2>")
            i += 1
        elif stripped.startswith("|"):
            start = i
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
            blocks.append(_table(lines[start:i]))
        elif stripped.startswith("- "):
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{_inline(lines[i].strip()[2:].strip())}</li>")
                i += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
        else:
            blocks.append(f"<p>{_inline(stripped)}</p>")
            i += 1
    return "\n".join(blocks)


def write_html(html_text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8", newline="\n")
    return path
