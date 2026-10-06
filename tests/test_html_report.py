"""The Markdown subset as HTML (the analyst's comment) and the UTF-8 writer."""

from __future__ import annotations

from pathlib import Path

from termo.operation.html_report import md_to_html, write_html


def test_headings_bold_and_paragraphs() -> None:
    text = "# TERMO — hoja\n\n## Fase actual\n\n**venta** desde 1999-01-04\n\nOtra línea"
    html = md_to_html(text)
    assert "<h1>TERMO — hoja</h1>" in html
    assert "<h2>Fase actual</h2>" in html
    assert "<p><strong>venta</strong> desde 1999-01-04</p>" in html
    assert "<p>Otra línea</p>" in html
    assert "**" not in html


def test_a_whole_bold_line_and_bold_inside_a_line() -> None:
    html = md_to_html("**CAMBIO DE FASE: a → b**\n\nFase al cierre: **venta**, desde ayer.")
    assert "<p><strong>CAMBIO DE FASE: a → b</strong></p>" in html
    assert "<p>Fase al cierre: <strong>venta</strong>, desde ayer.</p>" in html


def test_pipe_table_with_header_and_two_rows() -> None:
    text = "| Fase | Probabilidad |\n|---|---|\n| rally fuerte | 0.80 |\n| venta | 0.20 |"
    html = md_to_html(text)
    assert html.count("<table>") == 1 and html.count("</table>") == 1
    assert "<thead><tr><th>Fase</th><th>Probabilidad</th></tr></thead>" in html
    assert "<tbody>" in html and "</tbody>" in html
    assert "<tr><td>rally fuerte</td><td>0.80</td></tr>" in html
    assert "<tr><td>venta</td><td>0.20</td></tr>" in html
    assert html.count("<tr>") == 3
    assert "---" not in html


def test_transition_table_keeps_the_backslash_header() -> None:
    text = "| De \\ A | venta | Episodios con sucesor |\n|---|---|---|\n| venta | 50% | 2 |"
    html = md_to_html(text)
    assert "<th>De \\ A</th>" in html and "<td>50%</td>" in html


def test_bullet_list() -> None:
    html = md_to_html("Variables:\n\n- nivel corto: +0.12\n- pendientes: -0.03\n\nFin")
    assert "<ul><li>nivel corto: +0.12</li><li>pendientes: -0.03</li></ul>" in html
    assert html.count("<ul>") == 1
    assert "<p>Variables:</p>" in html and "<p>Fin</p>" in html


def test_text_is_escaped_and_the_arrow_and_accents_are_kept() -> None:
    html = md_to_html("a < b & c\n\n| x<y | r&d |\n|---|---|\n| <td> | → |\n\n- días → años")
    assert "a &lt; b &amp; c" in html
    assert "<th>x&lt;y</th><th>r&amp;d</th>" in html
    assert "<td>&lt;td&gt;</td><td>→</td>" in html
    assert "<li>días → años</li>" in html
    assert "<td><td>" not in html


def test_empty_lines_separate_paragraphs_and_produce_nothing() -> None:
    html = md_to_html("uno\n\n\n\ndos\n")
    assert html.count("<p>") == 2
    assert "<p></p>" not in html
    assert md_to_html("") == ""


def test_write_html_is_utf8_with_unix_newlines(tmp_path: Path) -> None:
    path = write_html("<!doctype html>\n<p>días →</p>\n", tmp_path / "out" / "reporte.html")
    assert path == tmp_path / "out" / "reporte.html"
    raw = path.read_bytes()
    assert raw.decode("utf-8") == "<!doctype html>\n<p>días →</p>\n"
    assert b"\r\n" not in raw
