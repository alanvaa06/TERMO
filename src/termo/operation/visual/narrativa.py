"""Headlines and bullets of the visual report: closed templates filled with the data.

Every sentence describes the past or the present; none uses the future or the
conditional (tests/test_visual_narrativa.py enforces it). Numbers come only from
DatosReporte and calculos.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from termo.operation.monthly import transitions
from termo.operation.visual import calculos
from termo.operation.visual.calculos import Cuadrante, Posicion
from termo.operation.visual.datos import DatosReporte
from termo.operation.visual.etiquetas import PLAZOS, bloque_es
from termo.validation.metrics import BP_PER_PERCENT

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
    "validacion",
)
NO_VALIDADO = (
    "Las explicaciones de esta semana no pasaron la validación de fidelidad del imitador"
)
CURVA_CORTA = "La curva tiene menos de 3 meses de historia"
ETAPAS = (("diagnostic", "diagnóstico"), ("holdout", "holdout"))  # validation stages, in order


@dataclass(frozen=True)
class Titular:
    texto: str
    vinetas: tuple[str, ...] = ()


def _capital(texto: str) -> str:
    return texto[:1].upper() + texto[1:]


def _enumerar(partes: Sequence[str]) -> str:
    """'a', 'a y b', 'a, b y c'."""
    return partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " y " + partes[-1]


def _movimiento(pb: float) -> str:
    if round(pb) == 0:
        return "no cambió"
    return f"{'subió' if pb > 0 else 'bajó'} {abs(pb):.0f} pb"


def titular_portada(fase: str, posicion: Posicion) -> str:
    """Today's length against the quartiles of the finished episodes of the phase."""
    base = f"La curva lleva {posicion.dias} {'día' if posicion.dias == 1 else 'días'} en {fase}"
    relativa, p25, p75 = posicion.relativa, posicion.p25, posicion.p75
    if relativa is None or p25 is None or p75 is None:
        return base
    cerrados = f"los episodios de {fase} cerrados"
    if relativa == "por encima":
        return f"{base}, más que el P75 de {cerrados} ({p75:.0f} días)"
    if relativa == "por debajo":
        return f"{base}, menos que el P25 de {cerrados} ({p25:.0f} días)"
    return f"{base}, entre el P25 y el P75 de {cerrados} ({p25:.0f} a {p75:.0f} días)"


def titular_10a(fase: str, inicio: str, cambio_pb: float) -> str:
    return f"El 10A {_movimiento(cambio_pb)} desde que empezó el episodio de {fase} ({inicio})"


def titular_acuerdo(acuerdo: float) -> str:
    return f"Imitador y fase coinciden en el {acuerdo:.0%} de los días del último año"


def titular_motor(bloque: str, aporte: float) -> str:
    """The sheet's first driver: the block with the largest absolute contribution."""
    return f"{_capital(bloque_es(bloque))} es el bloque que más pesa en la lectura ({aporte:+.2f})"


def titular_motores_tiempo(bloque: str, inicio: str, media: float) -> str:
    """`media` is the block's signed mean contribution since `inicio`."""
    return (
        f"Desde el {inicio}, {bloque_es(bloque)} es el bloque con mayor peso, "
        f"{media:+.2f} de promedio"
    )


def titular_curva(cambio_2y: float, cambio_10y: float) -> str:
    """The 2A-10A slope changed by d10 - d2 bp: positive steepens, negative flattens."""
    if calculos.paralelo(cambio_2y, cambio_10y):
        pendiente = "no cambió"
    else:
        cambio = cambio_10y - cambio_2y
        pendiente = f"{'se empinó' if cambio > 0 else 'se aplanó'} {abs(cambio):.0f} pb"
    return (
        f"En 3 meses el 2A {_movimiento(cambio_2y)} y el 10A {_movimiento(cambio_10y)}; "
        f"la pendiente 2A-10A {pendiente}"
    )


def titular_firma(fase: str, plazo: str | None, valor_pb: float | None) -> str:
    if plazo is None or valor_pb is None:
        return f"Sin cambios a 1 mes registrados para {fase}"
    return f"En {fase}, el {plazo} tiene el mayor cambio mediano a un mes ({valor_pb:+.0f} pb)"


def titular_episodios(
    fase: str, cuadrantes: Sequence[Cuadrante], n: int, total: int
) -> str:
    """`total` counts the CLOSED episodes of the phase; the open one is not history yet.

    `cuadrantes` are the most frequent quadrants, `n` episodes each: on a tie every one
    is named with its own count, as in titular_transicion.
    """
    if total == 0 or not cuadrantes:
        return f"Es el primer episodio de {fase} en la historia"
    if total == 1:
        return f"El único episodio de {fase} ya cerrado fue {cuadrantes[0].value}"
    verbo = "fue" if n == 1 else "fueron"
    partes = [f"{verbo} {cuadrantes[0].value}", *(f"{n} {c.value}" for c in cuadrantes[1:])]
    return f"{n} de {total} episodios de {fase} ya cerrados {_enumerar(partes)}"


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
    return f"De {total} episodios de {fase} ya cerrados, {_enumerar(partes)}"


def titular_macro(etiqueta: str, percentil: float, fecha_dato: str | None = None) -> str:
    """`fecha_dato` when the value predates the reading (the sheet's 'nota' says so)."""
    texto = f"{_capital(etiqueta)} está en el percentil {percentil * 100:.0f} a 10 años"
    return texto if fecha_dato is None else f"{texto} (dato del {fecha_dato})"


def titular_validacion(veredictos: Sequence[Mapping[str, object]], semanas: int) -> str:
    """The registered diagnostic and holdout verdicts (a missing stage is left out) and the
    weeks of shadow."""
    por_etapa = {str(v["stage"]): str(v["verdict"]) for v in veredictos}
    partes = [
        f"{nombre} {por_etapa[etapa].replace('-', ' ').upper()}"
        for etapa, nombre in ETAPAS
        if etapa in por_etapa
    ]
    sombra = f"{semanas} {'semana' if semanas == 1 else 'semanas'} de sombra"
    if not partes:
        return _capital(sombra)
    return f"{_capital(' y '.join(partes))}; {sombra}"


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
        dominante, media = calculos.bloque_dominante(historia, datos.bloques, actual["inicio"])
        texto_motor = titular_motor(str(primero["block"]), float(primero["contribution"]))
        texto_tiempo = titular_motores_tiempo(dominante, inicio, media)
    else:
        texto_motor = texto_tiempo = NO_VALIDADO

    curvas = calculos.curvas_pasadas(datos.curva, datos.fecha)
    if "hace 3m" in curvas.index:
        cambio = (curvas.loc["hoy"] - curvas.loc["hace 3m"]) * BP_PER_PERCENT
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
        texto_episodios = titular_episodios(fase, (), 0, 0)
    else:
        conteo = calculos.cuadrantes(suyos).value_counts()
        n_moda = int(conteo.max())
        modas = [c for c in Cuadrante if int(conteo.get(c, 0)) == n_moda]  # enum order
        texto_episodios = titular_episodios(fase, modas, n_moda, len(suyos))

    salida = transitions(datos.episodios, datos.nombres)[fase_n]

    macro = max(datos.hoja["macro"], key=lambda m: abs(float(m["percentil_10a"]) - 0.5))
    # the sheet's 'nota' is non-empty when the last value predates the reading date
    fecha_dato = str(macro["fecha_valor"]) if macro["nota"] else None
    texto_macro = titular_macro(str(macro["serie"]), float(macro["percentil_10a"]), fecha_dato)
    texto_acuerdo = titular_acuerdo(coincidencia)
    validacion = lectura["validation"]
    texto_validacion = titular_validacion(
        validacion["registered_verdicts"], int(validacion["sombra"]["semanas"])
    )

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
        "validacion": Titular(texto_validacion),
    }
