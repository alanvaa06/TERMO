---
Writer: Claude
---

# TERMO — Spec 3: herramienta descriptiva

**Fecha:** 2026-10-05
**Estado:** diseño aprobado en brainstorming; pendiente de revisión del spec escrito.
**Documentos base:** [spec 1](2026-10-01-nucleo-go-no-go-design.md) y [spec 2](2026-10-04-exp2-receta-tyccles-design.md), sus reportes ([`reports/go_no_go.md`](../../../reports/go_no_go.md), [`reports/exp2/go_no_go.md`](../../../reports/exp2/go_no_go.md)), TYCCLES (HSBC).

---

## 1. Qué cambia y por qué

Dos experimentos registrados (62 trials) dieron NO-GO a la misma pregunta: las fases no dicen del movimiento futuro del 10Y más que la inercia. El usuario decidió (2026-10-05) que TERMO siga como **herramienta solo descriptiva**: dice en qué fase está la curva hoy, con qué confianza y por qué. No anticipa.

Esto es una **decisión de producto nueva**, no un cambio retroactivo de criterio: los veredictos NO-GO quedan como están y la herramienta lleva escrito lo que no hace.

HSBC hace el mismo descargo para TYCCLES ("contemporaneous model... not designed to forecast"), pero publica gráficas del movimiento posterior condicionadas a que la fase continúe. **TERMO no publica nada condicionado al futuro.**

### Dentro de este spec

1. Modelo fijo de 3 fases (ya corrido en exp2).
2. Imitador XGBoost: confianza y explicación SHAP por bloques.
3. Criterios descriptivos registrados antes de abrir el holdout, y la prueba única en holdout.
4. Lectura semanal mínima (Markdown + JSON) con descargo fijo.

### Fuera (spec 4, después del holdout)

Reporte mensual con gráficas, tablas de transiciones y duraciones típicas, alertas, panel macro, modo sombra, operación.

---

## 2. Lo que se fija, y qué se sabía al fijarlo

Todo esto se eligió **mirando datos pre-holdout** (1988 → 2024-09). Se declara; por eso la evidencia que cuenta es la del holdout.

| Pieza | Valor | Origen |
|---|---|---|
| Datos | snapshot `2026-10-02`, 7 series, desde 1977-02-15 | specs 1 y 2 |
| Variables | receta de rangos, 139 variables, sin regla de colinealidad, arranque 504 días | spec 2 §4 |
| Motor | jump model discreto, **K = 3**, multa por variable **c = 3** (λ = 417) | trial `jm_k3_lam3` de exp2 |
| Reentrenamiento | ventana expansiva desde 1987-12-31, cada 26 semanas, lectura en línea | spec 1 |
| Nombres | fase 0 = **rally de la parte corta**, 1 = **rally de la parte larga**, 2 = **venta** | descripción hecha el 2026-10-05 |

Por qué K = 3: es el único K con mapa estable en las dos pruebas (S2 mitades 0.76, S1 0.91). K = 4 da el mapa de HSBC por intensidad, pero S2 = 0.37 y cambia con la multa. K = 2 es más estable pero se sospecha (no verificado) que es solo sube/baja.

Qué significan las fases (pre-holdout, cambios a 63 días): venta = 10Y +28 pb, 2Y +29 pb, curva sin cambio (47% de los días); rally largo = 10Y −29, 2Y −15, aplana 14 pb (33%); rally corto = 10Y −38, 2Y −56, empina 18 pb (20%). La volatilidad no distingue fases.

Una sola configuración: no hay rejilla ni selección en este spec.

---

## 3. Imitador y explicación

### 3.1 Imitador

- XGBoost multiclase (3 clases). Parámetros fijados aquí, sin afinar: 300 árboles, profundidad 4, tasa 0.05, submuestra 0.8, columnas por árbol 0.8, `tree_method=hist`, semilla 0, un hilo.
- En cada reentrenamiento: entrada = las 139 variables estandarizadas del día; objetivo = etiquetas del jump model **ajustadas en la ventana de entrenamiento** (con nombres ya alineados). Lee el bloque siguiente hasta el próximo reentrenamiento. Mismo corte que el jump model: sin look-ahead.
- No recibe la fase de ayer: imita desde las variables del día. Por eso discrepará sobre todo en transiciones (el jump model tiene inercia por la multa; el imitador no).

### 3.2 Confianza

- Probabilidad que el imitador da a cada fase. **Confianza** = la que da a la fase del jump model.
- Aviso de **baja confianza** si esa probabilidad < 0.6 o si la fase más probable del imitador no es la del jump model.
- Calibración (Brier por fase y tabla de fiabilidad en 5 tramos): se reporta, no decide.

### 3.3 SHAP por bloques

- `TreeExplainer` sobre el imitador vigente en la fecha, para la clase de la fase del jump model.
- Los 139 aportes (en log-odds, la escala interna del modelo) se suman en 6 bloques (la suma es exacta por aditividad):

| Bloque | Variables |
|---|---|
| Nivel corto | `d1_*`, `d2_*`, `d3_*` |
| Nivel medio | `d5_*`, `d7_*` |
| Nivel largo | `d10_*`, `d30_*` |
| Pendientes | `s12m5s_*`, `s3s10_*`, `s10s30_*` |
| Curvatura | `c5_*` |
| Volatilidad | `vol*` |

- Signo positivo = empuja hacia esa fase. Se listan además las 5 variables individuales con mayor aporte absoluto, como apéndice.
- **SHAP explica al imitador, no al mercado ni al jump model.** Solo vale si la fidelidad pasa (§4).

---

## 4. Criterios descriptivos

Todos contemporáneos. Se calculan dos veces con el mismo código: sobre pre-holdout (diagnóstico, contaminado por la elección) y sobre holdout (decide).

| # | Criterio | Regla | Pre-holdout conocido |
|---|---|---|---|
| D1 | Persistencia | duración mediana de cada fase ≥ 20 días | 54–82 |
| D2 | Coherencia de la venta | en fase 2, el 10Y sube a 63 días en ≥ 60% de los días | 80% |
| D3 | Coherencia de los rallies | en fases 0 y 1, el 10Y baja a 63 días en ≥ 60% de los días | 83% y 81% |
| D4 | Quién lidera | fase 0: cambio medio a 63 días de (10Y − 2Y) > 0; fase 1: < 0 | +18 y −14 pb |
| D5 | El mapa no cambia al reentrenar | ARI ≥ 0.6 entre las etiquetas del modelo congelado en el último corte previo y las del walk-forward, sobre los días evaluados | sin medir |
| D6 | Fidelidad del imitador | acierto balanceado ≥ 0.80 contra las etiquetas en línea del jump model | sin medir |

Reglas:

- Cambios a 63 días = diferencia simple de la serie cruda (no la suavizada de las variables).
- Una fase con menos de 40 días en el periodo evaluado es **no evaluable** para D1–D4 y tampoco cuenta en D6 (decisión del usuario, 2026-10-05, antes de registrar y sin haber visto el diagnóstico): tres días fallados de una fase que casi no aparece no deben decidir. El acierto del imitador por fase, incluidas las no evaluables, se reporta aparte sin decidir. Se reporta y no reprueba. Si las tres son no evaluables, el veredicto es NO APTO por falta de evidencia.
- D5 sobre pre-holdout: un jump model ajustado una sola vez con datos hasta 2014-12-31 (y nunca reentrenado) contra el walk-forward, en los días posteriores hasta 2024-09-30. Sobre holdout: ajustado una sola vez con todos los datos hasta 2024-09-30, contra el walk-forward en los días del holdout. Es el mismo mecanismo de "refit congelado" de spec 2.
- **Veredicto:** APTO si pasan todos los criterios evaluables; NO APTO si falla alguno.
- **El holdout solo se abre si el diagnóstico pre-holdout es APTO.** Si D5 o D6 fallan en pre-holdout, la herramienta ya no sirve y no se gasta el holdout.
- Los umbrales no se tocan después de ver el diagnóstico pre-holdout.

Lo que estos criterios no prueban (declarado antes de correr):

- D2–D4 son en parte por construcción: las fases se arman con rangos de esos mismos cambios. Comprueban que los nombres siguen valiendo en datos nuevos, no que haya información nueva.
- El 0.80 de D6 es juicio, no está calibrado.
- El holdout tiene ~2 años (~500 días, del orden de 8–12 episodios): prueba gruesa. Un APTO dice "no se rompió en dos años", no más.
- Nada aquí mide capacidad de anticipar. Esa pregunta está cerrada con NO-GO.

---

## 5. Bitácora y holdout

- Registro nuevo: `trials/desc/trials.jsonl`, etiquetas y probabilidades en `trials/desc/`, reportes en `reports/desc/`. Los registros de specs 1 y 2 no se tocan.
- Un `setup` + **1 trial** (`desc_k3`). Divulgación: 62 trials previos + 1 = **63**; ambos registros previos van en `prior_trial_logs`.
- Mismos amarres: configuración completa, hash del snapshot, identidad del código, versiones (se añaden `xgboost` y `shap` al entorno amarrado), SHA-256 de cada archivo de salida.
- Etapas: `register` → `run` (walk-forward pre-holdout del jump model y del imitador; diagnóstico D1–D6) → `report` (veredicto de diagnóstico) → `holdout` (una sola vez).
- **Holdout:** exige diagnóstico APTO registrado y aprobación explícita del usuario en ese momento. Todo lo que puede fallar se comprueba antes de marcarlo abierto. Política estricta ya decidida por el usuario: abierto sin resultado = perdido.
- El holdout de este spec es el mismo periodo que specs 1 y 2 reservaron (desde 2024-10-01); nunca se abrió. Tras este spec queda gastado.

---

## 6. Lectura semanal

Comando con fecha (`read --date AAAA-MM-DD`). Salida `reports/desc/readings/AAAA-MM-DD.md` y `.json`:

- Fase del jump model y su nombre; días hábiles en la fase; fecha de inicio del episodio.
- Probabilidad del imitador por fase; aviso de baja confianza si aplica.
- Los 3 bloques con mayor aporte absoluto, con signo; apéndice con 5 variables.
- Estado de validación: "diagnóstico pre-holdout: APTO/NO APTO" y, si existe, "holdout: APTO/NO APTO".
- Descargo fijo, siempre: *"TERMO describe la fase actual de la curva. No anticipa el 10Y: en 62 pruebas registradas las fases no superaron a la inercia."*

Reglas:

- Antes de que exista un resultado de holdout, solo se pueden leer fechas anteriores a 2024-10-01 (el cargador se niega a lo demás).
- La lectura de una fecha usa solo datos hasta esa fecha (modelo e imitador del último corte anterior).
- Ninguna cifra sobre movimientos posteriores a la fecha.
- La lectura se produce siempre, también con veredicto NO APTO: el comité recibe su hoja cada semana. Si el informe que gobierna (el holdout cuando tiene resultado; si no, el diagnóstico pre-holdout) no es APTO, la hoja abre con un aviso de "no validada" que nombra los criterios fallidos; los impulsores se muestran igual, y si falló D6 el encabezado de impulsores avisa que las explicaciones no están validadas.
- `read` amarra la configuración, el snapshot y el SHA-256 de cada archivo que carga, pero no el commit del código ni las versiones de librerías: un cambio de código o una actualización posterior sigue permitiendo leer esta bitácora. La lectura registra con qué commit se generó y cuál quedó registrado. Las demás etapas conservan el amarre completo.

---

## 7. Código

Se reusa: receta de rangos, pipeline, walk-forward, alineación de nombres, bitácora, amarres, cargador con candado de holdout.

Nuevo:

| Módulo | Responsabilidad |
|---|---|
| `surrogate/model.py` | ajustar el imitador en una ventana, dar probabilidades; walk-forward del imitador sobre los mismos cortes |
| `surrogate/explain.py` | SHAP de una fecha, suma por bloques, mapa bloque ← variable |
| `descriptive/criteria.py` | D1–D6 sobre un periodo; veredicto; regla de no evaluable |
| `descriptive/stages.py` | registrar, correr, reportar, holdout de este spec |
| `reading.py` | lectura de una fecha a Markdown y JSON |
| `configs/desc.yaml` | todo lo de §2–§4 como valores registrados |
| CLI | subcomandos de las etapas y `read` |

`pyproject.toml` añade `xgboost` y `shap`. Primera tarea del plan: prueba de humo de ambos en este entorno (Python 3.14, numpy 2.5); la descarga del instalador está verificada, que funcionen no.

### Pruebas (pytest)

- Imitador sin look-ahead: alterar datos después del corte no cambia el modelo ni las probabilidades hasta el corte; determinista con la misma semilla.
- Fidelidad alta en la curva sintética con fases plantadas; baja con etiquetas barajadas.
- SHAP: la suma de los 6 bloques más el valor base reproduce la salida del modelo para esa clase; cada variable cae en exactamente un bloque.
- Criterios: casos a mano para D1–D6, fase no evaluable, tres no evaluables = NO APTO.
- Etapas: holdout se niega sin diagnóstico APTO; nada falla después de marcarlo abierto por algo comprobable antes; una sola vez; registros previos intactos (hash).
- Lectura: se niega a fechas de holdout sin resultado; no usa datos posteriores a la fecha; contiene el descargo; ASCII en consola.

---

## 8. Riesgos y limitaciones

- Elección de K, multa y nombres hecha sobre datos vistos; solo el holdout es limpio, y es corto.
- El imitador puede no alcanzar 0.80 por diseño (no tiene inercia). Si falla, la alternativa registrada para un spec posterior es la explicación por centroides (exacta, sin imitador); no se cambia el umbral.
- 139 variables correlacionadas: SHAP reparte el aporte entre gemelas; por eso se reporta por bloque.
- La fase de venta no distingue bear flattener de bear steepener; la volatilidad no pesa.
- Limitaciones heredadas: DGS30 2002–2006, snapshot fijo (lecturas más allá de 2026-10-01 requieren nuevo snapshot y es operación de spec 4).
