"""Headlines and bullets of the visual report: closed templates filled with the data.

Every sentence describes the past or the present; none uses the future or the
conditional (tests/test_visual_narrativa.py enforces it). Numbers come only from
DatosReporte and calculos.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from termo.operation.monthly import transitions
from termo.operation.visual import calculos
from termo.operation.visual.calculos import Cuadrante, Posicion
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.etiquetas import bloque_es

CLAVES = (
    "portada",
    "tasa-10a",
    "imitador",
    "motores",
    "motores-tiempo",
    "curva",
    "firma",
    "episodios",
    "transiciones",
    "macro",
)
PLAZOS = {"DGS1": "1A", "DGS2": "2A", "DGS3": "3A", "DGS5": "5A", "DGS7": "7A", "DGS10": "10A",
          "DGS30": "30A"}
NO_VALIDADO = (
    "Las explicaciones de esta semana no pasaron la validación de fidelidad del imitador"
)
CURVA_CORTA = "La curva tiene menos de 3 meses de historia"


@dataclass(frozen=True)
class Titular:
    texto: str
    vinetas: tuple[str, ...] = ()


def _capital(texto: str) -> str:
    return texto[:1].upper() + texto[1:]


def _movimiento(pb: float) -> str:
    if round(pb) == 0:
        return "no cambió"
    return f"{'subió' if pb > 0 else 'bajó'} {abs(pb):.0f} pb"


def titular_portada(fase: str, posicion: Posicion) -> str:
    base = f"La curva está en {fase}: {posicion.dias} días"
    if posicion.relativa is None:
        return base
    return f"{base}, {posicion.relativa} del rango intercuartil histórico"


def titular_10a(fase: str, inicio: str, cambio_pb: float) -> str:
    return f"El 10A {_movimiento(cambio_pb)} desde que empezó el episodio de {fase} ({inicio})"


def titular_acuerdo(acuerdo: float) -> str:
    return f"Imitador y fase coinciden en el {acuerdo:.0%} de los días del último año"


def titular_motor(bloque: str, aporte: float) -> str:
    """The sheet's first driver: the block with the largest absolute contribution."""
    return f"{_capital(bloque_es(bloque))} es el bloque que más pesa en la lectura ({aporte:+.2f})"


def titular_motores_tiempo(bloque: str, inicio: str) -> str:
    return f"Desde {inicio}, el bloque de mayor aporte medio es {bloque_es(bloque)}"


def titular_curva(cambio_2y: float, cambio_10y: float) -> str:
    if calculos.paralelo(cambio_2y, cambio_10y):
        forma = "se movió en paralelo"
    else:
        forma = "se aplanó" if cambio_2y > cambio_10y else "se empinó"
    return (
        f"En 3 meses el 2A {_movimiento(cambio_2y)} y el 10A {_movimiento(cambio_10y)}: "
        f"la curva {forma}"
    )


def titular_firma(fase: str, plazo: str | None, valor_pb: float | None) -> str:
    if plazo is None or valor_pb is None:
        return f"Sin cambios a 1 mes registrados para {fase}"
    return (
        f"En {fase}, el plazo que más se mueve en un mes (mediana) es el {plazo} "
        f"({valor_pb:+.0f} pb)"
    )


def titular_episodios(fase: str, cuadrante: Cuadrante | None, n: int, total: int) -> str:
    """`total` counts the CLOSED episodes of the phase; the open one is not history yet."""
    if total == 0 or cuadrante is None:
        return f"Es el primer episodio de {fase} en la historia"
    if total == 1:
        return f"El único episodio de {fase} ya cerrado fue {cuadrante.value}"
    return f"{n} de {total} episodios de {fase} ya cerrados fueron {cuadrante.value}"


def titular_transicion(fase: str, total: int, proporciones: Mapping[str, float | None]) -> str:
    """Past frequency: of the `total` closed episodes, how many gave way to each phase.

    `proporciones` follows phase order; on a tie every tied phase is named, each with
    its own count, so no single episode reads as going to two phases.
    """
    validas = {nombre: p for nombre, p in proporciones.items() if p is not None}
    if total == 0 or not validas:
        return f"No hay episodios de {fase} ya cerrados"
    maxima = max(validas.values())
    destinos = [nombre for nombre, p in validas.items() if p == maxima]
    if total == 1:
        return f"El único episodio de {fase} ya cerrado dio paso a {destinos[0]}"
    k = round(maxima * total)
    cuenta = f"{k} ({maxima:.0%})"
    verbo = "dio" if k == 1 else "dieron"
    partes = [f"{cuenta} {verbo} paso a {destinos[0]}", *(f"{cuenta} a {d}" for d in destinos[1:])]
    lista = partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " y " + partes[-1]
    return f"De {total} episodios de {fase} ya cerrados, {lista}"


def titular_macro(etiqueta: str, percentil: float) -> str:
    return f"{_capital(etiqueta)} está en el percentil {percentil * 100:.0f} a 10 años"


def titulares(datos: DatosReporte) -> dict[str, Titular]:
    """One headline per block of the page (keys in CLAVES); the cover has three bullets."""
    lectura = datos.lectura
    fase_n = int(lectura["phase"])
    fase = datos.nombres[fase_n]
    actual = datos.episodios.iloc[-1]
    inicio = actual["inicio"].strftime("%Y-%m-%d")
    historia = datos.historia.loc[: datos.fecha]

    posicion = calculos.posicion_duracion(datos.episodios, fase_n, int(lectura["days_in_phase"]))
    coincidencia = calculos.acuerdo(historia, datos.nombres)
    if lectura["drivers_validated"]:
        primero = lectura["drivers"][0]  # ranked by absolute contribution, as the sheet
        dominante, _ = calculos.bloque_dominante(historia, datos.bloques, actual["inicio"])
        texto_motor = titular_motor(str(primero["block"]), float(primero["contribution"]))
        texto_tiempo = titular_motores_tiempo(dominante, inicio)
    else:
        texto_motor = texto_tiempo = NO_VALIDADO

    curvas = calculos.curvas_pasadas(datos.curva, datos.fecha)
    if "hace 3m" in curvas.index:
        cambio = (curvas.loc["hoy"] - curvas.loc["hace 3m"]) * 100.0
        curva = titular_curva(float(cambio["DGS2"]), float(cambio["DGS10"]))
    else:
        curva = CURVA_CORTA

    firma = calculos.firma_por_fase(datos.curva, historia["fase"], datos.nombres).loc[fase]
    firma = firma.dropna()
    if firma.empty:
        texto_firma = titular_firma(fase, None, None)
    else:
        plazo = str(firma.abs().idxmax())
        texto_firma = titular_firma(fase, PLAZOS.get(plazo, plazo), float(firma[plazo]))

    cerrados = datos.episodios.iloc[:-1]  # the last episode is the current, still open
    suyos = cerrados[cerrados["fase"] == fase_n]
    if suyos.empty:
        texto_episodios = titular_episodios(fase, None, 0, 0)
    else:
        cuadrantes = calculos.cuadrantes(suyos)
        moda = Cuadrante(cuadrantes.mode().iloc[0])
        n_moda = int((cuadrantes == moda).sum())
        texto_episodios = titular_episodios(fase, moda, n_moda, int(len(cuadrantes)))

    salida = transitions(datos.episodios, datos.nombres)[fase_n]

    macro = max(datos.hoja["macro"], key=lambda m: abs(float(m["percentil_10a"]) - 0.5))
    texto_macro = titular_macro(str(macro["serie"]), float(macro["percentil_10a"]))
    texto_acuerdo = titular_acuerdo(coincidencia)

    return {
        "portada": Titular(
            titular_portada(fase, posicion), (texto_motor, texto_acuerdo, texto_macro)
        ),
        "tasa-10a": Titular(titular_10a(fase, inicio, float(actual["cambio_10y_pb"]))),
        "imitador": Titular(texto_acuerdo),
        "motores": Titular(texto_motor),
        "motores-tiempo": Titular(texto_tiempo),
        "curva": Titular(curva),
        "firma": Titular(texto_firma),
        "episodios": Titular(texto_episodios),
        "transiciones": Titular(titular_transicion(fase, int(salida["total"]), salida["a"])),
        "macro": Titular(texto_macro),
    }
