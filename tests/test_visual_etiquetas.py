"""Spanish labels: the six SHAP blocks and every variable of the registered recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from termo.config import load_config
from termo.dataset import recipe_for
from termo.features.tyccles import TycclesRecipe
from termo.operation.visual.etiquetas import bloque_es, variable_es

REPO = Path(__file__).resolve().parents[1]


def test_the_level_blocks_say_movement_not_level() -> None:
    assert bloque_es("nivel corto") == "movimiento tramo corto (1A-3A)"
    assert bloque_es("nivel medio") == "movimiento tramo medio (5A-7A)"
    assert bloque_es("nivel largo") == "movimiento tramo largo (10A-30A)"
    assert bloque_es("pendientes") == "pendientes"
    assert bloque_es("curvatura") == "curvatura"
    assert bloque_es("volatilidad") == "volatilidad"


def test_an_unknown_block_is_refused() -> None:
    with pytest.raises(ValueError, match="nivel ultra"):
        bloque_es("nivel ultra")


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [
        ("d5_63_r252", "5A · cambio 3m · rango 1a"),
        ("d1_21_r126", "1A · cambio 1m · rango 6m"),
        ("d30_189_r252", "30A · cambio 9m · rango 1a"),
        ("s12m5s_42_r126", "pendiente 1s5s · cambio 2m · rango 6m"),
        ("s3s10_84_r252", "pendiente 3s10s · cambio 4m · rango 1a"),
        ("s10s30_63_r252", "pendiente 10s30s · cambio 3m · rango 1a"),
        ("c5_189_r252", "curvatura 2·5A−2A−10A · cambio 9m · rango 1a"),
        ("vol10_r252", "10A · volatilidad 21d · rango 1a"),
    ],
)
def test_variables_are_translated(nombre: str, esperado: str) -> None:
    assert variable_es(nombre) == esperado


def test_every_variable_of_the_registered_recipe_is_translated() -> None:
    recipe = recipe_for(load_config(REPO / "configs" / "desc2.yaml"))
    assert isinstance(recipe, TycclesRecipe)
    assert len(recipe.names) == 139
    traducidas = {variable_es(name) for name in recipe.names}
    assert len(traducidas) == 139  # no two variables share a label


@pytest.mark.parametrize("nombre", ["d5_64_r252", "d5_63_r100", "x5_63_r252", "vol10_r126x"])
def test_an_unknown_variable_is_refused(nombre: str) -> None:
    with pytest.raises(ValueError, match="variable"):
        variable_es(nombre)
