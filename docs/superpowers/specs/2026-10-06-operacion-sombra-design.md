---
Writer: Claude
---

# TERMO — Spec 4: operación en modo sombra, hoja semanal y ficha mensual

**Fecha:** 2026-10-06
**Estado:** diseño aprobado en brainstorming; pendiente de revisión del spec escrito.
**Documentos base:** [spec 3](2026-10-05-termo-descriptivo-design.md) (§9: registro `desc2`), [`TERMO-diseno-comite.md`](../../design/TERMO-diseno-comite.md) §5 paso 8 y §6, reportes de `desc2` ([diagnóstico](../../../reports/desc2/diagnostic.md), [holdout](../../../reports/desc2/holdout.md)).

---

## 1. Objetivo

Que el comité reciba cada semana una hoja con la fase actual de la curva, y cada mes una ficha completa, con trazabilidad total y sin ninguna afirmación predictiva. Y que las semanas nuevas se acumulen como evidencia limpia ("sombra") para validar los nombres de fase elegidos después del holdout.

### Dentro

1. Etapa `sombra`: snapshot semanal, reproducción de la historia, lectura, alerta, bitácora.
2. Hoja semanal en español (Markdown + JSON).
3. Ficha mensual en español.
4. Evaluación de sombra cada 26 semanas (criterios D1–D3, D5, D6 sobre los días nuevos).
5. Cuatro CSV.
6. Panel macro de contexto: prima por plazo y expectativa de Fed (proxy).

### Fuera

Correo/Slack, Cochrane–Piazzesi, gráficas, cualquier estadística de movimiento posterior a la lectura, reentrenamiento distinto del walk-forward registrado.

---

## 2. Modelo y registro que se operan

Todo sale del registro **`desc2`** (`configs/desc2.yaml`, `trials/desc2/`): jump model de 3 fases (0 = rally fuerte, 1 = rally moderado, 2 = venta), receta de rangos, imitador XGBoost, SHAP por bloques. Nada de eso cambia. Su holdout quedó gastado el 2026-10-05; los veredictos registrados son: diagnóstico APTO, holdout (ya visto) APTO en D1–D3, D5, D6, y el holdout original de `desc_k3` NO APTO por D4 (afirmación retirada). La hoja lo dice siempre.

---

## 3. Etapa `sombra` (semanal)

```
lunes: snapshot nuevo -> config y columnas registradas? -> recomputar todo
   -> la historia registrada se reproduce byte a byte? -> lectura del ultimo viernes
   -> alerta? -> bitacora de sombra -> hoja (es) + JSON -> CSV
```

### 3.1 Snapshot

- Mismo mecanismo de spec 1 (`take_snapshot`: descarga, validación de cobertura, manifiesto con SHA-256), con las 7 series de la curva más las 2 macro (§7). Directorio `data/snapshots/AAAA-MM-DD/`, versionado en git (≈1 MB por semana).
- Las series macro no entran al modelo: solo al panel. Su falta de cobertura no detiene la lectura; se anota.

### 3.2 Amarre: reproducir la historia

El código de spec 4 añade módulos, así que la identidad de código registrada en `desc2` ya no coincide. Se sustituye por una prueba más fuerte:

- Se verifica que `configs/desc2.yaml` tiene la huella registrada y que las columnas son las 139 registradas.
- Se recomputa todo (walk-forward, imitador, SHAP) con el snapshot nuevo hasta su último día.
- Los cinco archivos restringidos a fechas ≤ **2026-09-30** (último día del snapshot registrado) deben ser **byte-idénticos** a `trials/desc2/pre_holdout` + `trials/desc2/holdout` (hash contra la bitácora).
- Si coinciden: lectura normal. Si no: la lectura se produce igual, con la marca **"HISTORIA REVISADA"**, y la bitácora guarda qué días y qué archivos difieren. (Causa esperable: una revisión de FRED. No se "arregla": se declara.)

Esto prueba, cada semana, que el modelo y el código producen exactamente lo registrado sobre los datos registrados.

### 3.3 Lectura y alerta

- Fecha de lectura = último viernes ≤ último día del snapshot con datos completos; si el viernes falta (feriado), el último día hábil anterior de esa semana.
- Contenido = el de spec 3 §6 (lookup en los archivos recomputados) + panel macro + estado de sombra.
- **Alerta** si la fase de la lectura ≠ fase de la lectura anterior en la bitácora de sombra **y** confianza ≥ 0.6. Texto: "CAMBIO DE FASE: venta → rally moderado el 2026-10-10". Va arriba de la hoja y en `alertas.csv`. El envío lo hace el usuario.
- Primera lectura de sombra: la anterior es la última del holdout (2026-09-30), tomada de los archivos registrados.

### 3.4 Bitácora de sombra

`trials/desc2/sombra/sombra.jsonl`, solo-añadir. Un registro por corrida: fecha de corrida, hash del snapshot, fecha de lectura, lectura completa (JSON), hashes de los cinco archivos recomputados (guardados en `trials/desc2/sombra/<fecha>/`), resultado de la reproducción de la historia, alerta. Una corrida repetida para la misma fecha de lectura se registra como tal (no sobrescribe).

### 3.5 Costo

~15–25 min por corrida (recomputa todo). Aceptable para una vez por semana; no se optimiza en este spec.

---

## 4. Hoja semanal (español)

Archivo `reports/desc2/sombra/AAAA-MM-DD.md` (UTF-8, acentos permitidos; la consola sigue ASCII) y `.json`.

1. Título, fecha de lectura, fecha y hash del snapshot.
2. Alerta, si hay. Banda "NO VALIDADO" si el veredicto que gobierna no es APTO. Marca "HISTORIA REVISADA" si aplica.
3. **Fase actual** y desde cuándo (días hábiles). Confianza del imitador con la nota fija: *"el imitador es sobreconfiado: cuando dice 0.98 acierta ~89% (medido en holdout)"*. Probabilidad por fase.
4. **Qué la empuja**: 3 bloques con signo, 5 variables; aviso si la fidelidad falló.
5. **Contexto macro** (§7): prima por plazo 10 años y 2Y−FF: valor, percentil en 10 años, cambio en 21 días hábiles. Sin interpretación.
6. **Validación**: veredictos registrados, tabla por fase (holdout), semanas de sombra acumuladas y fecha de la próxima evaluación, nombres elegidos tras el holdout.
7. Descargo fijo: *"TERMO describe la fase actual de la curva. No anticipa la tasa a 10 años: en 62 pruebas registradas las fases no superaron a la inercia."*
8. Procedencia: identidad del código con que se generó.

---

## 5. Ficha mensual (`ficha --mes AAAA-MM`)

Archivo `reports/desc2/fichas/AAAA-MM.md` + `.json`, generada bajo demanda desde la bitácora de sombra, los archivos registrados y el snapshot más reciente. Secciones:

1. Fase al cierre del mes y desde cuándo; lecturas del mes (tabla: fecha, fase, confianza); alertas del mes.
2. Qué la empujó: promedio por bloque de las lecturas del mes.
3. **Duraciones históricas** por fase (mediana, rango intercuartil, episodios) y **tabla de transiciones** (de fase a fase, frecuencia), sobre 1988 → último día disponible, rotuladas: *"frecuencias del pasado, no pronóstico; no probadas en holdout"*.
4. Contexto macro: nivel, percentil, cambio en el mes.
5. Validación (como en la hoja) y "Lo que el modelo NO dice: hacia dónde irá la tasa ni qué posición tomar".
6. Descargo y procedencia.

No incluye "movimiento típico en esta fase" ni ninguna cifra condicionada a lo que pasa después de una lectura.

---

## 6. Evaluación de sombra (`evaluar-sombra`)

- Se puede correr cuando hay ≥ 26 semanas de sombra (configurable); el usuario la lanza.
- Entrada: etiquetas, probabilidades y curva de los **días posteriores a 2026-09-30** (tomados de la última corrida de sombra, hash-verificados).
- Criterios D1, D2, D3, D5, D6 con los mismos umbrales y la misma regla de "no evaluable" (`desc2.yaml`). D5 usa un modelo congelado con datos hasta 2026-09-30 contra el walk-forward.
- Salida: `reports/desc2/sombra_eval_<fecha>.md/.json`, registro en la bitácora de sombra. Veredicto APTO / NO APTO **de la sombra**; no reescribe ningún veredicto anterior. Es la validación limpia de los nombres de `desc2`.

---

## 7. Panel macro

| Serie | Fuente | Uso |
|---|---|---|
| Prima por plazo 10 años (Kim–Wright) | FRED `THREEFYTP10`, diaria desde 1990 | contexto |
| Fed funds efectiva | FRED `DFF` | proxy de expectativa: `DGS2 − DFF` |

- Sustituye al ACM del diseño (NY Fed lo publica en Excel fuera de FRED; mismo concepto, otra estimación). **Verificado en FRED el 2026-10-06:** `THREEFYTP10` diaria desde 1990-01-02, publicada con ~1 semana de retraso (último dato 2026-09-25); `DFF` diaria desde 1954 incluyendo fines de semana (se toma el valor del día hábil).
- Por serie: valor en la fecha de lectura, percentil dentro de los últimos 10 años (2,520 días), cambio en 21 días hábiles. Sin umbrales ni semáforos.
- Huecos: si una serie macro no tiene dato en la fecha, se usa el último disponible y se anota.

---

## 8. CSV (`exportar`; también al final de cada `sombra`)

En `reports/desc2/csv/`, separador coma, UTF-8, una cabecera con `snapshot_hash` y `generado`:

| Archivo | Una fila por | Columnas |
|---|---|---|
| `lecturas.csv` | lectura semanal de sombra | fecha, fase, nombre, días en fase, inicio del episodio, p_rally_fuerte, p_rally_moderado, p_venta, confianza, baja_confianza, bloque_1..3 con aporte, alerta, historia_revisada, validación, snapshot_hash |
| `historia_diaria.csv` | día hábil desde 1988 | fecha, fase, nombre, p0, p1, p2, suma SHAP por bloque (6), base, periodo (pre_holdout / holdout / sombra) |
| `episodios.csv` | episodio | inicio, fin, fase, nombre, días, cambio del 10Y y del 2Y dentro del episodio (pb), periodo |
| `macro.csv` | día hábil | fecha, prima_10a, percentil_10a, cambio_21d, dgs2_menos_ff, percentil, cambio_21d |
| `alertas.csv` | alerta | fecha, de, a, confianza |

---

## 9. Configuración

`configs/operacion.yaml` (no forma parte de la huella de `desc2`): series macro y sus nombres, umbral de alerta 0.6, ventana de percentil 2,520 días, cambio 21 días, mínimo de semanas para evaluar la sombra 26, textos fijos (descargo, nota de sobreconfianza). Se carga con el mismo patrón que las otras configs; claves desconocidas se rechazan.

---

## 10. Código

| Módulo | Responsabilidad |
|---|---|
| `operation/config.py` | `OperationConfig` + cargador |
| `operation/macro.py` | series macro desde el snapshot: valor, percentil, cambio |
| `operation/shadow.py` | etapa `sombra`: reproducción de la historia, lectura, alerta, bitácora `ShadowLog` |
| `operation/sheet_es.py` | hoja semanal en español (Markdown) |
| `operation/monthly.py` | ficha mensual: duraciones, transiciones, agregados del mes |
| `operation/exports.py` | los cinco CSV |
| `operation/shadow_eval.py` | evaluación de sombra |
| `op_cli.py` | subcomandos `sombra`, `ficha`, `evaluar-sombra`, `exportar` |

Se reusa: `take_snapshot`, `load_curve`, `prepare`, `analyse`, `restrict`, `analysis_bytes`/`hashes_of`, `build_reading`, criterios, `TrialLog` (patrón de bitácora append-only). `desc_cli` y `desc2.yaml` no cambian.

### Pruebas (pytest)

- Reproducción de la historia: snapshot sintético idéntico → "reproducida"; snapshot con un dato pasado alterado → "HISTORIA REVISADA" con los días listados, y la lectura se produce igual.
- Fecha de lectura: viernes; viernes feriado → jueves; snapshot que termina en miércoles → viernes anterior.
- Alerta: cambio de fase con confianza ≥ 0.6 → alerta; < 0.6 → sin alerta; misma fase → sin alerta; primera corrida compara contra la última del holdout.
- Bitácora de sombra solo-añadir; corrida repetida registrada, no sobrescrita.
- Hoja: UTF-8 con acentos, contiene descargo y nota de sobreconfianza, no contiene cifras posteriores a la fecha; consola ASCII.
- Ficha: transiciones suman 1 por fila; duraciones coinciden con `episodios.csv`; rótulo de "no pronóstico" presente.
- Macro: percentil y cambio a mano; hueco → último disponible y nota.
- CSV: columnas exactas; una fila por unidad; releíbles con pandas.
- Evaluación de sombra: con < 26 semanas se niega; con datos sintéticos produce los 5 criterios; no escribe en `trials/desc2/trials.jsonl`.

---

## 11. Riesgos y límites

- El imitador es sobreconfiado; la nota va en cada hoja.
- Los nombres de fase se eligieron tras el holdout; solo la evaluación de sombra los valida. Hasta entonces la hoja lo dice.
- Duraciones y transiciones históricas se leen como pronóstico aunque se rotulen; el rótulo es la defensa, y "movimiento típico" no existe en ningún documento.
- Recomputar todo cada semana cuesta 15–25 min; si FRED revisa datos, la historia registrada ya no se reproduce y cada hoja lo dirá hasta que se decida qué hacer (decisión del usuario, no automática).
- El proxy 2Y−FF no es la expectativa de mercado exacta (incluye prima por plazo corta).
- Series macro desde 1990: el percentil de 10 años está disponible desde 2000.
