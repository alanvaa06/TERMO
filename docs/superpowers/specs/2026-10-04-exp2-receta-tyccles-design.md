---
Writer: Claude
---

# TERMO — Spec 2: experimento 2, receta de datos de TYCCLES

**Fecha:** 2026-10-04
**Estado:** diseño aprobado en brainstorming (decisiones 1a, 2a, 3a); pendiente de revisión del spec escrito.
**Documentos base:** [spec 1](2026-10-01-nucleo-go-no-go-design.md), [`reports/go_no_go.md`](../../../reports/go_no_go.md) (veredicto NO-GO del núcleo), TYCCLES (HSBC, PDF en Descargas; líneas 366–383 del texto extraído), wiki del vault (TYCCLES, ECCLES, EMCCLES, DUSTIN).

---

## 1. Objetivo

El experimento 1 (spec 1) dio **NO-GO**: con 10 variables en z-score (PCA, velocidades, vol en logs) ninguna configuración del jump model alcanzó estabilidad 0.6 ni separó más que la inercia; las líneas base K-means con reentrenamiento dieron fases de 2–5 días. Conclusión registrada: el problema no parece ser el motor sino lo que se le da de comer.

HSBC (TYCCLES) obtiene fases persistentes con K-means sobre una receta de datos distinta: cambios a 1–9 meses convertidos en **rangos** (percentiles) de 6 meses y 1 año. Este spec prueba esa receta con **nuestras** pruebas, nuestros umbrales y nuestro registro, sin cambiar ningún criterio después de ver un resultado.

> **Pregunta:** ¿con la receta de datos de TYCCLES aparecen fases estables, persistentes, que separan el movimiento futuro del 10Y más que la inercia?

### Hipótesis declaradas antes de correr

- **H1 (diagnóstico, no criterio):** con rangos, las líneas base K-means con reentrenamiento dejan de tener fases de días y pasan la duración mínima. Si H1 falla, la persistencia de TYCCLES no viene de los datos.
- **H2:** al menos una configuración del jump model pasa todos los criterios bloqueantes.
- **H3:** el K-means congelado (una sola vez, al estilo HSBC) pasa todos los criterios bloqueantes.

Advertencia escrita de antemano: los rangos sobre cambios de meses son suaves por construcción, así que **estabilidad y duración pueden pasar "gratis"**. La prueba que decide de verdad es separación contra la inercia e independencia.

### Dentro de este spec

- Receta de variables TYCCLES adaptada a nuestras 7 series.
- Dos motores: jump model (walk-forward como spec 1) y K-means congelado.
- Mismas pruebas, umbrales, bitácora y amarres que spec 1. Registro nuevo, separado del de spec 1.

### Fuera de este spec

- XGBoost + SHAP (solo si este experimento da GO; sería spec 3).
- Todo lo que spec 1 dejó fuera (CJM, macro, reporte, operación).
- Réplica completa de TYCCLES con su evidencia de estudio de eventos (si este experimento da NO-GO, el usuario decide pasar a esa opción "b").

---

## 2. Flujo

```
snapshot 2026-10-02 (el mismo) -> cambios 1-9m suavizados -> rangos 6m y 1a
   -> recorte y z-score -> { jump model walk-forward | K-means congelado }
   -> pruebas -> veredicto por motor -> veredicto del experimento
```

---

## 3. Datos (M1)

Idénticos a spec 1: mismo snapshot `data/snapshots/2026-10-02/` (mismo hash), mismas 7 series (DGS1, 2, 3, 5, 7, 10, 30), inicio 1977-02-15, holdout desde 2024-10-01 **cerrado**. No se descarga nada nuevo.

Diferencia con HSBC: TYCCLES arranca en 1967 (datos de Bloomberg); nosotros en 1977 (inicio de DGS30 en FRED). Declarada.

---

## 4. Variables (M2): la receta

### 4.1 Lo que dice TYCCLES (verificado en el PDF)

- Cambios de tasa a horizontes de 1, 2, 3, 4, 6 y 9 meses, en "múltiples puntos" de la curva.
- Esos cambios se evalúan como grandes o pequeños "relative to recent history" con rangos de 6 meses y 1 año.
- Pendientes 12m5s, 3s10s, 10s30s y curvatura en 5Y; sus cambios a los mismos horizontes.
- Volatilidad realizada en varios plazos.
- Todos los cambios suavizados (Base Jumping). 102 insumos.

**No dice** qué plazos usa, qué ventana de vol, ni si los rangos sustituyen a los cambios crudos. Nuestra reconstrucción (no verificada): 4 plazos × 6 horizontes × 2 rangos = 48, 4 medidas de curva × 6 × 2 = 48, vol en 6 plazos = 6 → 102. Cuadra, pero es inferencia.

### 4.2 Nuestra adaptación (decisiones 1a y 2a)

Todo en puntos base. Horizontes en días hábiles: `H = {21, 42, 63, 84, 126, 189}`.

| Bloque | Serie base | Cambio | Rangos | Cuenta |
|---|---|---|---|---|
| Nivel | cada uno de los 7 plazos | 6 horizontes, suavizado | 126 y 252 días | 7 × 6 × 2 = 84 |
| Curva | 5Y−1Y (12m5s), 10Y−3Y (3s10s), 30Y−10Y (10s30s), 2·5Y−2Y−10Y (curvatura 5Y) | 6 horizontes, suavizado | 126 y 252 días | 4 × 6 × 2 = 48 |
| Vol | desviación estándar de los cambios diarios en 21 días, cada uno de los 7 plazos | — | 252 días | 7 |

**Total: 139 variables.** Entran **solo los rangos** (2a); ningún cambio crudo ni nivel.

- **Suavizado:** `smoothed_change(serie, h, delta)` de spec 1 (promedio de los cambios a h−delta, h y h+delta días, todas las ventanas terminan hoy) con `delta = h // 4` → 5, 10, 15, 21, 31, 47. Desviación respecto a spec 1 (63 → 10 allí, 15 aquí): declarada.
- **Rango causal:** rango percentil de hoy dentro de la ventana de los últimos `w` días **incluido hoy** (`rolling(w).rank(pct=True)`, método promedio). Solo mira el pasado: sin look-ahead por construcción. Antes de rankear, la serie se redondea a 6 decimales (en pb): FRED publica 2 decimales, y sin redondeo el ruido de punto flotante rompe los empates (8% de las celdas, hasta 0.02 de rango, medido en datos sintéticos de 2 decimales durante la revisión del código, antes de registrar).
- **Sin regla de colinealidad** (1a): HSBC metió 102 insumos correlacionados; podarlos cambia la receta. Queda declarado que las variables de nivel adyacentes están muy correlacionadas y pesarán más en la distancia. `collinearity_max` queda en la config pero no se aplica (`apply_collinearity_rule: false`).
- **Preproceso:** recorte a ±3σ y z-score con estadísticas de la ventana de entrenamiento, como spec 1. Los rangos ya viven en (0, 1], así que el recorte es inerte; el z-score se mantiene para que la escala de λ sea comparable entre experimentos.
- **Arranque (burn-in):** 504 días hábiles (cambio más largo 189+47 = 236 días, más rango de 252). Primera fecha con variables ≈ 1979-02.
- **Inercia:** signo del cambio suavizado a 63 días (`delta` 10) del **promedio de los 7 plazos** (proxy de nivel; en spec 1 era el factor L del PCA, que ya no existe aquí). Desviación declarada.
- **Nombres de fase:** fase 0 = menor media del cambio diario del 10Y en el primer ajuste, como spec 1.

### 4.3 Lo que NO se replica de TYCCLES

Plazos exactos, ventana de vol, "vol relativa", datos desde 1967, K=4 fijo, XGBoost, SHAP, estudio de eventos. Nuestra prueba sustituye a su evidencia; no la reproduce.

---

## 5. Motores (M3)

### 5.1 Jump model, walk-forward (como spec 1)

Primera ventana hasta 1987-12-31, reentrenamiento cada 26 semanas, ventana expansiva, lectura en línea, alineación de nombres por Hungarian. Grilla `K ∈ {2, 3, 4, 5}`.

**Penalización por variable.** El costo del JM suma distancias al cuadrado sobre `p` variables; con `p = 139` en vez de 10, el mismo λ pesa 14 veces menos. Se declara la grilla como **penalización por variable** `c ∈ {0.5, 1.2, 3, 8, 20, 50}` y `λ = c · p`. Con `p = 10` reproduce exactamente la grilla de spec 1 (5, 12, 30, 80, 200, 500); con `p = 139` da 69.5, 167, 417, 1112, 2780, 6950. Es la misma grilla en unidades comparables, no una grilla nueva.

### 5.2 K-means congelado (decisión 3a)

Al estilo HSBC: se ajusta **una sola vez** con datos desde el arranque hasta **1997-12-31** (pipeline y centroides), y etiqueta 1998-01 → 2024-09 por centroide más cercano, sin reentrenar nunca. `K ∈ {2, 3, 4, 5}`. Ids `kmf_k{K}`.

- 20 años dentro / ~27 fuera (≈1,390 semanas de lectura). HSBC usó 50 dentro / 9 fuera; congelar en 2007 dejaría ~870 semanas y poca potencia.
- **Estabilidad = S2** (mitades de toda la muestra pre-holdout, como `halves_ari`). S1 no existe: no hay reentrenamientos. Regla distinta por motor, fijada aquí.
- Las pruebas de separación, independencia, duración y PBO se calculan sobre 1998-01 → 2024-09.

### 5.3 Líneas base

- `kmeans_k{K}` con reentrenamiento (como spec 1): sirven para H1.
- `inertia` sobre el periodo de cada motor.

### 5.4 Grilla registrada

24 JM + 4 K-means reentrenado + 4 K-means congelado + inercia + setup = **34 registros = 33 trials + setup**. Con los 29 de spec 1, el proyecto lleva **62 trials registrados**; el reporte lo dice.

---

## 6. Bitácora y amarres

Igual que spec 1 (registro antes de correr, amarre a configuración completa, hash del snapshot, identidad del código, versiones, SHA-256 de etiquetas), con:

- **Registro nuevo y separado:** `trials/exp2/trials.jsonl`, etiquetas en `trials/exp2/labels/`, reporte en `reports/exp2/`. El registro de spec 1 no se toca.
- **Consecuencia del amarre al código:** cualquier cambio de código invalida las etapas pendientes del registro de spec 1 (`verify_binding` las rechaza). Spec 1 ya publicó su reporte; su holdout sigue cerrado y así queda. Se documenta.
- **Divulgación obligatoria en el reporte:** número de trials de este registro, de registros anteriores (`prior_trial_logs` en la config, leídos y contados), y la frase de que los datos pre-holdout ya fueron vistos por el experimento 1. El holdout no se abre en este experimento salvo GO y aprobación explícita del usuario (regla de spec 1).
- Hipótesis H1–H3 van en el registro `setup`.

---

## 7. Selección y criterios

### 7.1 Por familia, no en una sola bolsa

La estabilidad del JM (promedio de S1 y S2, con S1 ≈ 0.85 por construcción) no es comparable con la del K-means congelado (solo S2). Por eso **cada motor tiene su propio ganador y su propia tabla de criterios**:

- Familia JM: ganador por puntaje `estabilidad × separación` entre las configuraciones elegibles (como spec 1), FTIC como segunda opinión sobre K al λ del ganador, alternativa más simple si es elegible.
- Familia K-means congelado: ganador por el mismo puntaje con estabilidad = S2.

### 7.2 Criterios (sin cambios respecto a spec 1)

Bloqueantes: estabilidad ≥ 0.6; separación vs inercia con IC bootstrap 95% por bloques cuyo límite inferior > 0 (nivel de azar por volteo de signos); independencia p < 0.01 (χ² con todos los desplazamientos circulares); duración mediana ≥ 20 días en toda fase (elegibilidad). No bloqueantes: PBO ≤ 0.05 (por familia; con 4 configs en la familia congelada el PBO es casi ciego, se declara), S2 ≥ 0.6.

Horizonte que decide: 20 días; 65 días reportado.

### 7.3 Veredicto del experimento

- **GO** si al menos una familia pasa todos los bloqueantes; el reporte nombra cuál y, si ambas, prefiere la de mejor límite inferior de separación.
- **NO-GO** si ninguna. Entonces el usuario decide la opción "b" (réplica completa de TYCCLES con XGBoost y SHAP) o cerrar la línea. **No se ajusta nada sobre estos resultados.**

### 7.4 Potencia

La potencia de spec 1 se midió con 1,900 semanas simuladas. La familia congelada lee ~1,390. Se repitió la misma simulación (`sim_b.py`, modo `power1390`, 400 réplicas por fila, datos sintéticos, 2026-10-04) **antes** de registrar nada. Probabilidad de que el criterio de separación (límite inferior > 0) detecte un efecto real de tamaño η²:

| Semanas | η² = 0.01 | η² = 0.02 | η² = 0.05 |
|---|---|---|---|
| 1,900 (spec 1) | 15% | 40% | 94% |
| 1,390 (congelado) | 8% | 28% | 83% |

Con separación **e** independencia a la vez: 7%, 23%, 81% a 1,390 semanas (13%, 37%, 94% a 1,900). Lectura: para la familia congelada, un efecto moderado (2% de la varianza del movimiento a 4 semanas) se detecta ~1 de cada 4 veces. Un NO-GO de esa familia es débil como evidencia en contra; un GO sigue siendo igual de exigente.

---

## 8. Cambios de código (resumen; el plan detalla)

El código de spec 1 se conserva; se generaliza en los puntos mínimos:

1. **Receta de variables como objeto intercambiable.** `FittedPipeline` recibe una receta (`pca` = la de spec 1, `tyccles` = la nueva). La receta sabe calcular variables crudas causales y el cambio de nivel para la inercia. `fit_pipeline`, `build_refits`, `halves_ari` y `prepare` pasan por la receta; nada más cambia.
2. **`features/tyccles.py`:** cambios suavizados por plazo y por medida de curva, vol realizada, rangos causales, nombres explícitos (`d{plazo}_{h}_r{w}`, `s3s10_{h}_r{w}`, `c5_{h}_r{w}`, `vol{plazo}_r252`), proxy de nivel.
3. **Config:** `configs/exp2.yaml`. `CoreConfig` crece con `feature_set`, parámetros de la receta, `apply_collinearity_rule`, `jump_penalty_per_feature`, `frozen_train_end`, `prior_trial_logs`. `configs/core.yaml` sigue cargando igual (valores por defecto = comportamiento de spec 1).
4. **Motor congelado:** un solo "refit" con corte en `frozen_train_end` y lectura hasta el final; `evaluate_config` acepta la familia `kmeans_frozen` (S1 = no aplica, estabilidad = S2).
5. **Registro y reporte por familia;** divulgación de trials totales; `--trials-dir` y `--reports-dir` en el CLI (por defecto los de spec 1).
6. **Holdout final** acepta la familia ganadora (no solo JM). Sigue siendo una sola corrida con aprobación.

Prueba de tiempo antes de correr: con `p = 139` el JM tarda más por ajuste. Se mide en datos **sintéticos** del mismo tamaño (no en los reales) para estimar la duración de la corrida; si supera ~6 h, se registra igual y se corre en terminal con monitor, como en spec 1.

---

## 9. Pruebas del código (pytest)

Además de las 184 existentes (que deben seguir pasando):

- Rangos causales: cambiar el futuro de la serie no cambia el rango de hoy; el rango de hoy es el esperado en un caso a mano; con ventana `w`, las primeras `w−1` filas son NaN.
- Receta TYCCLES: 139 columnas con nombres esperados; sin valores no finitos tras el arranque; cambios suavizados coinciden con `smoothed_change` directo; curvatura y pendientes con signo correcto en una curva construida.
- Penalización por variable: con `p = 10` reproduce la grilla de spec 1.
- Motor congelado: una sola fecha de corte; etiquetas posteriores no cambian si se añaden datos al final (sin reentrenamiento); estabilidad = S2.
- Registro: 34 registros, hipótesis en `setup`, `prior_trial_logs` contados; registro de spec 1 intacto (hash del archivo antes y después).
- Reporte: tablas por familia; veredicto GO si una familia pasa; divulgación presente en el markdown.
- Config: `core.yaml` carga con los valores por defecto (su huella sí cambia, porque los campos nuevos entran con sus defaults; el registro de spec 1 ya quedó cerrado por la identidad del código, así que no importa).

---

## 10. Riesgos y limitaciones (declaradas antes de correr)

- Persistencia "gratis" por rangos suaves (ver hipótesis). Un GO que solo pase estabilidad y duración no existe: la separación decide.
- 139 variables correlacionadas: la distancia la dominan los cambios de nivel; las fases pueden ser "sube/baja" en disfraz. Si es así, la inercia debería empatar la separación y el criterio lo detecta.
- Reconstrucción de los 102 insumos es inferencia; nuestra receta no es TYCCLES exacto.
- El K-means congelado en 1997 usa centroides de la era 1979–1997 para leer 2008–2024; si falla, no refuta la versión de HSBC (50 años dentro).
- PBO con 4 configuraciones en la familia congelada es casi ciego.
- Segunda mirada a los mismos datos pre-holdout: toda la familia de pruebas tiene ya dos oportunidades; el reporte lo dice y el holdout queda cerrado.
- Limitaciones heredadas de spec 1 (DGS30 2002–2006, S1 alto por construcción, cero cuenta como "baja", etc.).
