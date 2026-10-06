"""The weekly deliverable as one HTML page (no network, no scripts).

The sheet and the monthly report are written in a small Markdown subset: `#`/`##`
headings, `**bold**`, pipe tables with a `|---|` separator row, `- ` bullets and blank-line
separated paragraphs. That subset, and only that, is converted here; any other line is a
paragraph. Text is escaped, so a `<` in a cell stays a `<`; accents and the `→` are kept.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from pathlib import Path

BOLD = re.compile(r"\*\*(.+?)\*\*")
TABLE_SEPARATOR = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")

STYLE = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2933;
  background: #ffffff; margin: 0; padding: 2rem 1rem; line-height: 1.5; }
main { max-width: 900px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 1rem; }
section { margin: 2.5rem 0; padding-top: 1rem; border-top: 2px solid #d9dee3; }
section > h2.titulo { font-size: 1.3rem; color: #52606d; margin: 0 0 1rem; }
section h1 { font-size: 1.4rem; margin: 1rem 0; }
section h2 { font-size: 1.15rem; margin: 1.5rem 0 0.5rem; }
p { margin: 0.5rem 0; }
ul { margin: 0.5rem 0; padding-left: 1.5rem; }
table { border-collapse: collapse; margin: 0.75rem 0; width: 100%; font-size: 0.95rem; }
th, td { border: 1px solid #d9dee3; padding: 0.35rem 0.6rem; text-align: left; }
th { background: #e4e7eb; }
tbody tr:nth-child(even) { background: #f5f7fa; }
footer { margin-top: 3rem; color: #7b8794; font-size: 0.85rem; }
"""


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


def render_html(sections: Sequence[tuple[str, str]], title: str, generated_at: str) -> str:
    """A self-contained HTML5 page in Spanish: one <section> per (heading, Markdown) pair."""
    parts = [
        "<!doctype html>",
        '<html lang="es">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{STYLE}</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    for heading, markdown in sections:
        parts += [
            "<section>",
            f'<h2 class="titulo">{html.escape(heading)}</h2>',
            md_to_html(markdown),
            "</section>",
        ]
    parts += [
        f"<footer>Generado el {html.escape(generated_at)}.</footer>",
        "</main>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


def write_html(html_text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8", newline="\n")
    return path
