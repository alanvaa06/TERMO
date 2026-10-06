"""Headlines and bullets of the visual report: closed templates filled with the data.

Every sentence describes the past or the present; none uses the future or the
conditional (tests/test_visual_narrativa.py enforces it). Numbers come only from
DatosReporte and calculos.
"""

from __future__ import annotations

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
    return (
        f"{_capital(bloque_es(bloque))} es el bloque de mayor aporte a la lectura ({aporte:+.2f})"
    )


def titular_motores_tiempo(bloque: str, inicio: str) -> str:
    return f"Desde {inicio}, el bloque de mayor aporte medio es {bloque_es(bloque)}"


def titular_curva(cambio_2y: float, cambio_10y: float) -> str:
    diferencia = cambio_2y - cambio_10y
    if abs(diferencia) < calculos.TOLERANCIA_PARALELO_PB:
        forma = "se movió en paralelo"
    else:
        forma = "se aplanó" if diferencia > 0 else "se empinó"
    return (
        f"En 3 meses el 2A {_movimiento(cambio_2y)} y el 10A {_movimiento(cambio_10y)}: "
        f"la curva {forma}"
    )


def titular_firma(fase: str, plazo: str, valor_pb: float) -> str:
    return (
        f"En {fase}, el plazo que más se mueve en un mes (mediana) es el {plazo} "
        f"({valor_pb:+.0f} pb)"
    )


def titular_episodios(fase: str, cuadrante: Cuadrante, n: int, total: int) -> str:
    return f"{n} de {total} episodios de {fase} fueron {cuadrante.value}"


def titular_transicion(
    fase: str, siguiente: str | None, proporcion: float | None, total: int
) -> str:
    if total == 0 or siguiente is None or proporcion is None:
        return f"No hay episodios de {fase} con sucesor"
    return f"Tras {fase}, el {proporcion:.0%} de los episodios siguió con {siguiente}"


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
    fila = historia.iloc[-1]
    motor = max(datos.bloques, key=lambda b: float(fila[b]))
    dominante, _ = calculos.bloque_dominante(historia, datos.bloques, actual["inicio"])

    curvas = calculos.curvas_pasadas(datos.curva, datos.fecha)
    if "hace 3m" in curvas.index:
        cambio = (curvas.loc["hoy"] - curvas.loc["hace 3m"]) * 100.0
        curva = titular_curva(float(cambio["DGS2"]), float(cambio["DGS10"]))
    else:
        curva = "La curva aún no tiene 3 meses de historia"

    firma = calculos.firma_por_fase(datos.curva, historia["fase"], datos.nombres).loc[fase]
    plazo = str(firma.abs().idxmax())

    cuadrantes = calculos.cuadrantes(datos.episodios)[datos.episodios["fase"] == fase_n]
    moda = Cuadrante(cuadrantes.mode().iloc[0])
    total_episodios = int(len(cuadrantes))
    n_moda = int((cuadrantes == moda).sum())

    salida = transitions(datos.episodios, datos.nombres)[fase_n]
    validos = {k: v for k, v in salida["a"].items() if v is not None}
    siguiente = max(validos, key=lambda k: validos[k]) if validos else None

    macro = max(datos.hoja["macro"], key=lambda m: abs(float(m["percentil_10a"]) - 0.5))
    texto_macro = titular_macro(str(macro["serie"]), float(macro["percentil_10a"]))
    texto_acuerdo = titular_acuerdo(coincidencia)

    return {
        "portada": Titular(
            titular_portada(fase, posicion),
            (titular_motor(motor, float(fila[motor])), texto_acuerdo, texto_macro),
        ),
        "tasa-10a": Titular(titular_10a(fase, inicio, float(actual["cambio_10y_pb"]))),
        "imitador": Titular(texto_acuerdo),
        "motores": Titular(titular_motor(motor, float(fila[motor]))),
        "motores-tiempo": Titular(titular_motores_tiempo(dominante, inicio)),
        "curva": Titular(curva),
        "firma": Titular(titular_firma(fase, PLAZOS.get(plazo, plazo), float(firma[plazo]))),
        "episodios": Titular(titular_episodios(fase, moda, n_moda, total_episodios)),
        "transiciones": Titular(
            titular_transicion(
                fase,
                siguiente,
                None if siguiente is None else validos[siguiente],
                int(salida["total"]),
            )
        ),
        "macro": Titular(texto_macro),
    }
