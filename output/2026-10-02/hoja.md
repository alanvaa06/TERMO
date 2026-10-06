# TERMO — hoja semanal del 2026-10-02

Snapshot del 2026-10-06 (hash a0faba4b93e5)

## Fase actual

**venta** desde 2025-12-18 (199 días hábiles)

Confianza del imitador: 1.00

El imitador es sobreconfiado: cuando dice 0.98 acierta cerca del 89% (medido en holdout). La confianza es su probabilidad, no una probabilidad calibrada.

| Fase | Probabilidad |
|---|---|
| rally fuerte | 0.00 |
| rally moderado | 0.00 |
| venta | 1.00 |

## Qué la empuja

- nivel medio: +2.70
- nivel largo: +0.72
- nivel corto: +0.68

Variables con mayor contribución:

- d5_63_r252: +0.91
- d7_84_r252: +0.50
- c5_189_r252: +0.30
- s10s30_63_r252: +0.24
- s10s30_84_r252: +0.17

## Contexto macro

| Serie | Valor | Percentil (10 años) | Cambio (21 días hábiles) | Nota |
|---|---|---|---|---|
| prima por plazo 10 anos (Kim-Wright) | 1.02 | 100% | 0.18 | último dato disponible: 2026-09-25 |
| DGS2 - fed funds efectiva | 0.95 | 92% | 0.19 |  |

## Validación

- Veredictos registrados: diagnostic APTO, holdout APTO
- Holdout: APTO

| Fase | Días | Evaluable | Duración mediana (días) | Dirección | Acierto |
|---|---|---|---|---|---|
| rally fuerte | 19 | no | 19.0 | 0.89 | 0.58 |
| rally moderado | 198 | sí | 198.0 | 0.71 | 0.72 |
| venta | 282 | sí | 141.0 | 0.91 | 0.96 |

Semanas de sombra: 1

Próxima evaluación de sombra en 25 semanas

Nombres de fase elegidos tras el holdout; los valida solo la sombra.

TERMO describe la fase actual de la curva. No anticipa la tasa a 10 años: en 62 pruebas registradas las fases no superaron a la inercia.

Generado con código bb1c653b7deb000fc821ce263fe17dba6c7764c1.6a5c57e0ac867a298aebe67f11d48e96347936aa.0bc700641939a01b2d9632b6a65563ca6117cdbc
