"""Spanish labels for the report: the SHAP blocks and the variables of the TYCCLES recipe.

Only the report renames; configs/desc2.yaml keeps its registered names. The three
"nivel" blocks measure recent CHANGES of the yields, ranked, not the level of rates,
so the report calls them "movimiento".
"""

from __future__ import annotations

import re

BLOQUES: dict[str, str] = {
    "nivel corto": "movimiento tramo corto (1A-3A)",
    "nivel medio": "movimiento tramo medio (5A-7A)",
    "nivel largo": "movimiento tramo largo (10A-30A)",
    "pendientes": "pendientes",
    "curvatura": "curvatura",
    "volatilidad": "volatilidad",
}
HORIZONTES: dict[str, str] = {
    "21": "1m",
    "42": "2m",
    "63": "3m",
    "84": "4m",
    "126": "6m",
    "189": "9m",
}
VENTANAS: dict[str, str] = {"126": "6m", "252": "1a"}
PLAZOS: dict[str, str] = {
    "DGS1": "1A",
    "DGS2": "2A",
    "DGS3": "3A",
    "DGS5": "5A",
    "DGS7": "7A",
    "DGS10": "10A",
    "DGS30": "30A",
}
MEDIDAS: dict[str, str] = {
    "s12m5s": "pendiente 1s5s",
    "s3s10": "pendiente 3s10s",
    "s10s30": "pendiente 10s30s",
    "c5": "curvatura 2·5A−2A−10A",
}
CAMBIO = re.compile(r"^(?P<serie>d\d+|s12m5s|s3s10|s10s30|c5)_(?P<h>\d+)_r(?P<w>\d+)$")
VOLATILIDAD = re.compile(r"^vol(?P<plazo>\d+)_r(?P<w>\d+)$")


def bloque_es(nombre: str) -> str:
    """'nivel medio' -> 'movimiento tramo medio (5A-7A)'."""
    try:
        return BLOQUES[nombre]
    except KeyError:
        raise ValueError(f"bloque desconocido: {nombre}") from None


def variable_es(nombre: str) -> str:
    """'d5_63_r252' -> '5A · cambio 3m · rango 1a'; an unknown pattern is an error."""
    if found := CAMBIO.match(nombre):
        serie = found["serie"]
        sujeto = f"{serie[1:]}A" if serie.startswith("d") else MEDIDAS[serie]
        horizonte = _buscar(HORIZONTES, found["h"], nombre)
        ventana = _buscar(VENTANAS, found["w"], nombre)
        return f"{sujeto} · cambio {horizonte} · rango {ventana}"
    if found := VOLATILIDAD.match(nombre):
        ventana = _buscar(VENTANAS, found["w"], nombre)
        return f"{found['plazo']}A · volatilidad 21d · rango {ventana}"
    raise ValueError(f"variable desconocida: {nombre}")


def _buscar(tabla: dict[str, str], clave: str, nombre: str) -> str:
    try:
        return tabla[clave]
    except KeyError:
        raise ValueError(f"variable desconocida: {nombre}") from None
