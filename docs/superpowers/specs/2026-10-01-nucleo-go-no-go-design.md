---
Writer: Claude
---

# TERMO — Spec 1: núcleo go/no-go

**Fecha:** 2026-10-01
**Estado:** diseño aprobado en brainstorming; pendiente de revisión del spec escrito.
**Documentos base:** [`TERMO-diseno-comite.md`](../../design/TERMO-diseno-comite.md), [`TERMO-diseno-tecnico.md`](../../design/TERMO-diseno-tecnico.md).

---

## 1. Objetivo

Responder una sola pregunta antes de construir el resto de TERMO:

> **¿Existen fases estables y distintas en la curva del Tesoro, detectables con un jump model, que digan algo más que la inercia simple?**

La salida es un veredicto **go / no-go** con su tabla de criterios. Si es no-go, no se construye SHAP, panel macro ni reporte.

### Dentro de este spec

- M1: datos de FRED con snapshot reproducible.
- M2: las 12 variables diarias.
- M3: jump model discreto con reentrenamiento walk-forward.
- M8: pruebas de estabilidad, separación y líneas base; bitácora; control de suerte; holdout.

### Fuera de este spec (specs posteriores)

- Probabilidades y bandas de confianza (CJM).
- Surrogate XGBoost + SHAP y atribución por centroides.
- Contexto histórico condicional (duraciones, transiciones).
- Panel macro (ACM, futuros, Cochrane-Piazzesi).
- Reporte mensual, alertas, shadow, monitoreo de deriva.
- Selección de variables (SJM), variante asimétrica de volatilidad, línea base HMM.

---

## 2. Flujo

```
FRED -> snapshot con hash -> PCA -> 12 variables -> recorte y z-score
     -> jump model (walk-forward) -> pruebas -> veredicto go/no-go
```

Cada caja es un módulo con una interfaz clara y sus pruebas en `pytest`.

---

## 3. Datos (M1)

| Tema | Especificación |
|---|---|
| Series | `DGS1`, `DGS2`, `DGS3`, `DGS5`, `DGS7`, `DGS10`, `DGS30` |
| Inicio | 1977-02-15 (primer dato de `DGS30`) |
| Fuente | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`, **una serie por petición** |
| Unidades | Tasas en porcentaje; los cambios se expresan en puntos base |
| Feriados | Se elimina todo día en que falte al menos uno de los 7 plazos. No se interpola ni se arrastra |
| Snapshot | `data/snapshots/<fecha>/<ID>.csv` + `manifest.json` con SHA-256 por archivo y fecha de descarga. Se versiona en git |
| Holdout | Fechas desde **2024-10-01** inclusive |

**Por qué una serie por petición.** Verificado: al pedir varias series juntas, FRED ignora el rango de fechas para todas menos la primera.

**Guardia de holdout.** El cargador de datos rechaza cualquier fecha desde 2024-10-01 salvo que se llame con la bandera explícita de evaluación final (§8.8). Es la única puerta al holdout.

**Validación en la frontera.** Al descargar: columnas esperadas, fechas crecientes sin duplicados, valores entre 0 y 25, serie no vacía. Al cargar: el hash del archivo coincide con el manifiesto.

**Cobertura.** La muestra conjunta debe empezar a más tardar 7 días después del inicio y no tener huecos de más de 10 días entre fechas consecutivas. Un hueco en una sola serie acortaría la muestra sin avisar. Se revisa al cargar y también sobre toda la descarga al tomar el snapshot, holdout incluido: solo mira qué días tienen dato, y un hueco descubierto después se descubriría con el holdout ya abierto.

### Datos verificados (2026-10-01, en filas del CSV de FRED)

| Serie | Hallazgo |
|---|---|
| `DGS3MO` | Empieza 1981-09-01. **Fuera del modelo** |
| `DGS2` | Empieza 1976-06-01 |
| `DGS30` | Empieza 1977-02-15. Tiene valores diarios en 2002-2006 |
| `DGS20` | Sin datos de 1987-01 a 1993-09. **Fuera del modelo** |

No se verificó el inicio de `DGS3`, `DGS5` ni `DGS10`; se asume 1962. La validación de descarga lo confirmará.

### Limitación conocida: la de 30 años en 2002-2006

`DGS30` salta +17 pb el 2002-02-19 y -16 pb el 2006-02-09, con la de 10 y la de 20 años casi planas. Son las fechas que FRED da como suspensión y regreso de la serie. Entre ambas, la serie queda unos 7 a 8 pb por debajo de la de 20 años (medido solo en los extremos).

**Decisión:** se usa cruda, sin excluir días ni ajustar el nivel. Los dos días aportan cerca de 0.1% de la varianza de la serie (estimado, no calculado). Queda documentado como limitación.

---

## 4. Variables (M2)

Se calculan a diario. Son 12.

| # | Grupo | Variable |
|---|---|---|
| 1 | Forma | Pendiente $S_t$ (puntaje PCA en nivel) |
| 2 | Forma | Curvatura $C_t$ (puntaje PCA en nivel) |
| 3 | Velocidad | Cambio suavizado de $L$ a 21 días |
| 4 | Velocidad | Cambio suavizado de $L$ a 63 días |
| 5 | Velocidad | Cambio suavizado de $S$ a 21 días |
| 6 | Velocidad | Cambio suavizado de $S$ a 63 días |
| 7 | Velocidad | Cambio suavizado de $C$ a 63 días |
| 8 | Nervio | $\log \sigma^{(60)}$ de la tasa a 5 años |
| 9 | Aceleración | $\log(\sigma^{(20)}/\sigma^{(60)})$ en 5 años |
| 10 | Aceleración | $\log(\sigma^{(60)}/\sigma^{(120)})$ en 5 años |
| 11 | Aceleración | $\log(\sigma^{(20)}/\sigma^{(60)})$ en 2 años |
| 12 | Aceleración | $\log(\sigma^{(20)}/\sigma^{(60)})$ en 10 años |

**El nivel $L_t$ no entra al agrupamiento.** Con tasas de 15% en 1981 y 0.5% en 2020, el modelo agruparía por "era de tasas". $L$ solo alimenta las velocidades 3 y 4.

### 4.1 PCA

- Se ajusta sobre la **covarianza de los cambios diarios** de los 7 plazos, solo con la ventana de entrenamiento. Queda congelado hasta el siguiente reentrenamiento.
- Puntajes en nivel: $X_t = \mathbf p_k^\top(\mathbf y_t - \bar{\mathbf y}_{\text{train}})$ para $k = 1, 2, 3$ ($L$, $S$, $C$).
- Convención de signo, aplicada tras cada ajuste:

| Componente | Regla |
|---|---|
| $L$ | Carga del plazo de 10 años positiva |
| $S$ | Carga de 30 años menos carga de 1 año, positiva |
| $C$ | Carga de 5 años menos el promedio de las cargas de 1 y 30 años, positiva |

### 4.2 Velocidad

Cambio a $h$ días hábiles: $\Delta_h X_t = X_t - X_{t-h}$. Suavizado con ventanas vecinas:

$$\widetilde\Delta_h X_t = \tfrac13\big(\Delta_{h-\delta}X_t + \Delta_h X_t + \Delta_{h+\delta}X_t\big)$$

con $(h, \delta) = (21, 5)$ y $(63, 10)$.

### 4.3 Volatilidad

$\sigma^{(hl)}_{j,t}$ es la raíz de la media exponencial de los cambios diarios al cuadrado (en pb) del plazo $j$, con vida media $hl$ y pesos normalizados. Solo usa datos hasta $t$.

Se usan logaritmos para que la era Volcker no domine: pasar de 4 a 8 pb pesa igual que pasar de 10 a 20.

### 4.4 Preproceso

1. Recorte de cada variable a ±3 desviaciones estándar (`DataClipperStd` de `jumpmodels`).
2. z-score (`StandardScalerPD` de `jumpmodels`).

Los límites, medias y desviaciones salen **solo de la ventana de entrenamiento**.

### 4.5 Arranque y colinealidad

- **Arranque:** se descartan los primeros 252 días hábiles de variables. La primera foto completa cae a inicios de 1978.
- **Colinealidad:** se revisa **una sola vez**, con la primera ventana de entrenamiento. Si dos variables tienen $|\rho| > 0.8$, se elimina la de número mayor en la tabla. El conjunto resultante queda fijo para todo el spec y se anota en la bitácora antes de correr la grilla.

### 4.6 El pipeline se reajusta completo en cada reentrenamiento

PCA, recorte y z-score se ajustan con la ventana de entrenamiento vigente y con ellos se recalculan las variables de todas las fechas que ese modelo va a leer.

---

## 5. Motor (M3)

| Tema | Especificación |
|---|---|
| Modelo | `jumpmodels.jump.JumpModel(cont=False)` |
| Arranques | `n_init=10`, `random_state=0` |
| Frecuencia | Datos diarios |
| Ventana | Expansiva desde el inicio |
| Primera ventana | Hasta el último día hábil de 1987 |
| Reentrenamiento | Cada 26 semanas, hasta 2024-09-30 |
| Lectura | `predict_online` con parámetros congelados |
| Lectura semanal | Estado del último día hábil de cada semana |

**Lectura en línea.** Para el bloque de 26 semanas que sigue a cada reentrenamiento, se corre `predict_online` sobre toda la secuencia, desde la primera fecha hasta el final del bloque, y se conserva solo el bloque. El estado en el día $t$ depende únicamente de datos hasta $t$ (verificado en el código de la librería).

### 5.1 Nombres de fase consistentes

Cada reentrenamiento puede permutar los índices de las fases. Se usa un solo mecanismo para todos los modelos (JM y líneas base):

1. **Primer modelo:** las fases se ordenan por el promedio del cambio diario de la 10 años dentro de la muestra, de menor a mayor. La fase 0 es la de mayor rally.
2. **Modelos siguientes:** se comparan las etiquetas dentro de muestra del modelo nuevo y del anterior (ya alineado) sobre las fechas de entrenamiento del anterior. La permutación que maximiza la coincidencia se obtiene con el algoritmo húngaro (`scipy.optimize.linear_sum_assignment`).

### 5.2 Grilla

| Hiperparámetro | Valores |
|---|---|
| $K$ | 2, 3, 4, 5 |
| $\lambda$ | 5, 12, 30, 80, 200, 500 |

Total: **24 configuraciones**. Referencia: el ejemplo de la librería usa $\lambda = 50$ con 9 variables diarias.

---

## 6. Bitácora

Regla del proyecto: **registrar antes de mirar**.

- Archivo `trials/trials.jsonl`, solo se agregan líneas.
- **Registro previo** (antes de correr): `trial_id`, fecha y hora, hipótesis, configuración completa, hash del snapshot, commit del código.
- **Registro de resultado** (después): `trial_id`, métricas, estado (`kept` / `discarded`), motivo, ruta de la serie de etiquetas (`trials/labels/<trial_id>.csv`) y el SHA-256 de ese archivo. El reporte recalcula sus criterios a partir de esos archivos, así que rechaza cualquiera que no coincida con su hash.
- Un trial que se cae a medias no deja resultado: queda pendiente y el mismo comando lo reintenta.
- El código **se niega** a guardar un resultado sin registro previo, o un segundo resultado para el mismo `trial_id`.
- **Amarre.** El registro de configuración guarda cuatro cosas: la configuración completa (fechas, horizontes, umbrales, semillas), el hash del snapshot, la identidad del código y las versiones de las librerías que calculan (numpy, pandas, scipy, scikit-learn, jumpmodels, Python). Las etapas de correr, reportar y abrir el holdout se niegan a trabajar si alguna cambió. Registrar exige código commiteado.
- **Identidad del código.** Es el hash de contenido que git da a `src/`, `configs/` y `pyproject.toml`, leído del repositorio desde donde corre el paquete. No cambia al commitear datos, bitácora o reportes, ni al reescribir el historial sin tocar el código; sí cambia con cualquier cambio de código o configuración.
- **Reportes.** Cada reporte generado se agrega a la bitácora con su veredicto y sus criterios. Un reporte nuevo no sustituye al anterior: queda al lado.
- **Holdout.** La apertura y el resultado quedan en la bitácora.

Las 24 configuraciones, las líneas base, los supuestos de FTIC y el conjunto de variables se registran antes de la primera corrida.

---

## 7. Selección de $K$ y $\lambda$

1. Para cada configuración se corre el walk-forward completo (§5).
2. Se descarta toda configuración con alguna fase de duración mediana menor a 20 días hábiles, o con alguna fase nunca visitada fuera de muestra.
3. Entre las restantes gana la de mayor puntaje:

$$\mathcal J = \text{estabilidad} \times \text{separación}_{4\text{ sem}}$$

La separación es el $\eta^2$ **en exceso** definido en §8.2.

Solo compiten las configuraciones con estabilidad positiva y separación positiva: el producto de dos negativos sería positivo. Si ninguna lo cumple, el veredicto es no-go.

4. **FTIC como segunda opinión sobre $K$** (§7.1). Si FTIC elige un $K$ distinto, el candidato pasa a ser $(\min(K^*, K_{\text{FTIC}}), \lambda^*)$. Si ese candidato pasa el filtro de duración y los criterios de no-go de §8.7, es el modelo final. Si no, se mantiene $(K^*, \lambda^*)$ y la discrepancia queda en el reporte.

### 7.1 FTIC

Criterio de Fan-Tang adaptado a jump models (Cortese, Kolm, Lindström, *AStA* 2026). Para cada configuración, con un solo ajuste sobre toda la muestra previa al holdout:

$$\text{FTIC} = \frac1T\Big[\text{WCSS} - \text{WCSS}_S + a_T\, M\Big] + 2\big[\log K - \log \bar K\big]$$

| Símbolo | Significado |
|---|---|
| WCSS | Suma de distancias al cuadrado de cada foto a su centroide |
| $\text{WCSS}_S$ | Lo mismo para el modelo saturado: $\bar K = 6$, $\lambda = 0$ |
| $a_T$ | $\log(\log T)\cdot\log p$, con $T$ días y $p$ variables |
| $M$ | $K(p + \delta_0) + K_0(\delta - \delta_0)$ |
| $\delta$ | Número de cambios de fase del modelo |

Supuestos previos, registrados en bitácora: $K_0 = 3$ y $\delta_0 = T/40$ (fase promedio de 40 días hábiles). Se excluye todo modelo con $\delta > 0.4\,T$.

$K_{\text{FTIC}}$ es el $K$ de menor FTIC **entre las configuraciones con $\lambda = \lambda^*$**.

**Por qué FTIC no elige $\lambda$.** Sin selección de variables, minimizar FTIC equivale a correr el jump model con $\lambda \approx a_T K_0 / 2$, cerca de 8 para esta muestra. Queda fijado por fórmula, no por los datos (derivación propia, sin probar). Además, FTIC trata cada día como independiente y nuestras variables son ventanas traslapadas.

---

## 8. Pruebas (M8)

Todas se calculan con lecturas **fuera de muestra** (1988 a 2024-09-30), salvo donde se indique.

### 8.1 Estabilidad

| Medida | Cómo |
|---|---|
| $S_1$ | ARI entre cada par de reentrenamientos consecutivos, sobre las fechas de entrenamiento del más antiguo. Se reporta media y mínimo |
| $S_2$ | Un modelo entrenado en la primera mitad de la muestra previa al holdout y otro en la segunda, cada uno con su propio pipeline. Ambos etiquetan toda la muestra con `predict`. ARI entre los dos etiquetados |

**Estabilidad** = promedio simple de $\bar S_1$ y $S_2$. Se reportan los dos por separado.

Advertencia: $S_1$ sale alto casi por construcción, porque dos ventanas consecutivas comparten más del 95% de los datos. $S_2$ es la prueba exigente. Si $S_2 < 0.6$ el reporte lo marca "a revisar" aunque el promedio pase.

### 8.2 Separación

- Lecturas semanales fuera de muestra.
- Objetivo: cambio de la tasa a 10 años de $t$ a $t+h$, en pb, con $h$ = 4 semanas (20 días hábiles) y 13 semanas (65 días hábiles).
- Se excluye toda lectura cuyo horizonte termine en el holdout.
- **Efecto crudo:** $\eta^2 = SS_{\text{entre}} / SS_{\text{total}}$.
- **Separación = $\eta^2$ en exceso:** el $\eta^2$ crudo menos su nivel de azar. El nivel de azar es el $\eta^2$ promedio que logran las mismas fases cuando a los movimientos se les voltea el signo al azar, por bloques de 26 semanas (2,000 sorteos, semilla fija).
- **Medición de comparación:** el exceso calculado con el nivel de azar anterior (las fases desplazadas en el tiempo). Se registra y se reporta para cada modelo y línea base. **Nunca decide**: ni el puntaje, ni los criterios, ni el holdout.
- **Intervalos:** bootstrap por bloques móviles de 26 semanas, 2,000 remuestreos, percentiles 2.5 y 97.5.

**Por qué en exceso.** Fases persistentes explican algo de la varianza de una serie autocorrelacionada por pura suerte, y más fases explican más. Medido con ruido puro al prototipar: cinco fases sin relación con las tasas le "ganaban" a dos fases sin relación el 25% de las veces, cuando lo nominal es 2.5%. Sin corrección, tanto la elección de $K$ como el criterio contra la inercia favorecen a los modelos con más fases.

**Por qué voltear signos y no desplazar las fases.** Una fase rara que coincide con las semanas más volátiles explica varianza por suerte: en esas semanas la tasa se mueve mucho hacia cualquier lado. Desplazar las fases rompe ese vínculo con la volatilidad y resta de menos. Voltear signos deja el tamaño de cada movimiento junto a su fase y borra solo la dirección. Medido con fases sin información de dirección (1,900 semanas, movimientos de 4 semanas traslapados):

| | Desplazar fases | Voltear signos |
|---|---|---|
| Separación inflada (debería ser 0) | +0.005 a +0.010 | +0.0003 a +0.0017 |
| El criterio contra la inercia pasa por suerte (nominal 2.5%) | 2% a 4% | 0.5% a 1.5% |
| Detecta una señal real | 94% | 90% |

Con una señal real muy fuerte el nivel de azar por volteo sube un poco, así que la medida es conservadora.

### 8.3 Independencia

Tabla fase × signo del cambio a 4 semanas, estadístico $\chi^2$. El valor p se obtiene desplazando circularmente la serie de fases por **todos** los desplazamientos posibles y contando cuántos igualan o superan al observado. Así se respeta la autocorrelación.

No se excluyen los desplazamientos pequeños. Medido al prototipar: con una zona de exclusión de 26 semanas la prueba rechaza el 4% de las veces cuando debería rechazar el 1%; con todos los desplazamientos rechaza el 0.7%.

El valor p más chico posible es 1 entre el número de semanas. Con pocos episodios de fase la prueba tiene poca potencia: en la curva sintética de prueba (unos 8 episodios plantados en 192 semanas) no bajó de 0.15.

### 8.4 Duración

Rachas de etiquetas diarias consecutivas iguales en la serie fuera de muestra, ya alineada y concatenada. Se reporta la mediana por fase.

### 8.5 Líneas base

| Línea base | Definición |
|---|---|
| Sin fases | Un solo grupo; $\eta^2 = 0$ |
| Inercia | Dos grupos según el signo de $\widetilde\Delta_{63} L$ |
| K-means | El mismo pipeline y walk-forward con $\lambda = 0$ y el mismo $K$ |

K-means se ajusta con `sklearn.cluster.KMeans`: minimiza el mismo objetivo que el jump model con multa cero, y con `jumpmodels` ese ajuste tarda más de un minuto por corrida.

La comparación contra la inercia usa un **bootstrap pareado**: los mismos bloques remuestreados para TERMO y para la inercia. El intervalo de la diferencia se corrige por la diferencia de sus niveles de azar (§8.2), para que más fases no den ventaja.

### 8.6 Control de suerte: PBO y $N$ efectivo

El ganador se elige mirando el movimiento futuro de la 10 años. Hay que medir cuánto de su ventaja es suerte.

**PBO por CSCV:**

1. Las lecturas semanales fuera de muestra se parten en 16 bloques contiguos.
2. Para cada una de las $\binom{16}{8} = 12{,}870$ combinaciones: 8 bloques son "dentro", 8 son "fuera".
3. Se calcula $\eta^2_{4\text{ sem}}$ de cada una de las 24 configuraciones en cada lado, hayan pasado o no el filtro de duración.
4. Se toma la mejor "dentro" y se mira su posición relativa "fuera".
5. PBO = fracción de combinaciones en que queda en la mitad inferior.

**$N$ efectivo:** agrupamiento jerárquico de las 24 series de etiquetas con distancia $1 - \text{ARI}$ y corte en 0.2. Se reporta 24 y $N_{\text{ef}}$.

Advertencia: si muchas configuraciones dan casi lo mismo, PBO sale cerca de 0.5 aunque las fases sean reales. Por eso su falla lleva a podar, no a cancelar.

PBO ordena las configuraciones por $\eta^2$ crudo. La corrección de azar de §8.2 no se puede recalcular en cada una de las 12,870 particiones a un costo razonable. Efecto medido: con una grilla que mezcla distintos $K$, PBO sale más bajo de lo debido (0.30 con ruido puro, contra 0.53 con todos los $K$ iguales). Nunca bajó de 0.15 con ruido, lejos del umbral de 0.05.

Los empates cuentan a medias: configuraciones con etiquetas idénticas dan PBO de 0.5, ni 0 ni 1.

### 8.7 Criterios de aceptación

Los umbrales son los propuestos en el diseño técnico §8.7. **El comité aún debe ratificarlos.**

| Criterio | Umbral | Si falla |
|---|---|---|
| Estabilidad (§8.1) | $\ge 0.6$ | **No-go** |
| Separación a 4 semanas de TERMO menos la de la inercia (ambas en exceso) | Intervalo de 95% de la diferencia por encima de 0 | **No-go** |
| Independencia (§8.3) | $p < 0.01$ | **No-go** |
| Duración mediana | $\ge 20$ días hábiles en todas las fases | La configuración se descarta |
| PBO | $\le 0.05$ | Podar la grilla y repetir, como trials nuevos |
| Holdout | Separación a 4 semanas de TERMO $\ge$ la de la inercia | Revisión antes de seguir |

**Si ninguna configuración pasa el filtro de duración: no-go.**

### 8.8 Evaluación final en holdout

- Se corre **una sola vez**, después de congelar el modelo final y anotarlo en bitácora.
- Requiere la bandera explícita y deja un registro `holdout_opened`. El código se niega a correrla si ese registro ya existe.
- Antes de marcar el holdout como abierto se comprueba todo lo que puede fallar: que el último reporte **de la bitácora** sea un go, que configuración, snapshot, código y librerías sean los registrados, que los archivos del snapshot coincidan con su manifiesto, que no haya huecos en ningún tramo (mirando solo fechas) y que el holdout tenga al menos 120 días. Un error de ruta no gasta la única apertura.
- El resultado también se agrega a la bitácora.
- El walk-forward continúa con reentrenamientos cada 26 semanas dentro del holdout.
- Con unas 104 semanas habrá pocas fases. El reporte da el número de semanas y de episodios, y advierte que el resultado es ruidoso.

---

## 9. Salidas

| Archivo | Contenido |
|---|---|
| `reports/go_no_go.md` | Veredicto, tabla de criterios con valor y pasa/falla, modelo elegido, discrepancia con FTIC, limitaciones |
| `reports/go_no_go.json` | Lo mismo, legible por máquina |
| `trials/trials.jsonl` | Bitácora |
| `trials/labels/` | Serie de etiquetas de cada trial |

Toda salida a consola es ASCII: `->`, `[ok]`, `[x]`.

---

## 10. Estructura del repositorio

```
TERMO/
├── configs/core.yaml        # series, fechas, grilla, umbrales, supuestos FTIC
├── data/snapshots/          # CSV de FRED + manifest.json
├── src/termo/
│   ├── data/                # descarga, snapshot, cargador con guardia de holdout
│   ├── features/            # PCA, velocidad, volatilidad, pipeline
│   ├── regime/              # interfaz de modelo (JM y K-means), alineación de fases
│   ├── validation/          # métricas, bloques, walk-forward, estabilidad,
│   │                        # FTIC, PBO, bitácora
│   ├── dataset.py           # datos compartidos por todas las configuraciones
│   ├── experiment.py        # evaluación de una configuración
│   ├── selection.py         # elección de K y lambda
│   ├── report.py            # criterios, veredicto, archivos de reporte
│   ├── core.py              # las cinco etapas
│   └── cli.py               # línea de comandos
├── trials/
├── reports/
├── tests/
└── pyproject.toml
```

**Dependencias:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `jumpmodels`, `pyyaml`, `pytest`.

---

## 11. Pruebas del código (pytest)

| Área | Qué se prueba |
|---|---|
| Sin look-ahead | Alterar datos posteriores a $t$ no cambia ninguna variable en $t$ |
| Ajuste solo en entrenamiento | PCA, recorte y z-score no cambian si se alteran datos fuera de la ventana |
| PCA | La convención de signo se cumple tras cada ajuste |
| Velocidad y volatilidad | Valores exactos sobre series sintéticas simples |
| Jump model | Recupera fases plantadas en una serie sintética |
| Alineación | Una permutación conocida de etiquetas se deshace |
| Walk-forward | Los cortes no se traslapan y ninguno toca el holdout |
| Métricas | ARI, $\eta^2$ y PBO contra casos con respuesta conocida |
| FTIC | Valor exacto en un caso calculado a mano |
| Bitácora | Rechaza resultados sin registro previo y resultados duplicados |
| Guardia de holdout | El cargador rechaza fechas del holdout sin la bandera |
| Datos | La validación rechaza CSV mal formados y hashes que no coinciden |

---

## 12. Riesgos y limitaciones

| Riesgo | Tratamiento |
|---|---|
| `jumpmodels` 0.1.1 (enero 2025) con Python 3.14 | Verificado al escribir el plan: funciona con numpy 2.5, pandas 3.0 y scikit-learn 1.9. Queda como prueba permanente |
| Los criterios son exigentes con pocos episodios | En la curva sintética con fases plantadas el veredicto fue no-go: la lectura en línea reconoce cada fase con retraso y la inercia es una línea base fuerte. Un no-go con datos reales es un resultado posible y legítimo |
| Las velocidades están en pb: los episodios de los 80 pueden definir "venta extrema" | Recorte a ±3 desviaciones; se revisa en el reporte qué años pueblan cada fase |
| La de 30 años en 2002-2006 es de otra construcción | Se usa cruda; documentado (§3) |
| $S_1$ inflado por datos compartidos | Se reporta $S_2$ aparte y se marca si queda bajo 0.6 |
| PBO alto con configuraciones parecidas | Falla lleva a podar; se reporta $N$ efectivo |
| Holdout corto | El criterio de holdout lleva a revisión, no a no-go automático |
| La evidencia del jump model viene de acciones | Las pruebas de §8 son el criterio, no la literatura |
| Tiempo de cómputo | Sin medir. Estimado grueso: del orden de una hora para la grilla completa |

### Desviaciones respecto al diseño técnico

1. La tasa a 3 meses sale; la de 30 años entra al PCA; no hay variable 10s30s aparte.
2. El nivel $L$ no entra al agrupamiento.
3. Volatilidad en logaritmos y recorte a ±3 desviaciones.
4. Sin embargo de 126 días entre entrenamiento y evaluación: el modelo no aprende de datos futuros. Solo se excluyen horizontes que cruzan al holdout.
5. Estabilidad sin bootstrap por bloques de años.
6. FTIC como segunda opinión sobre $K$, no como primera etapa.
7. Línea base HMM fuera.
8. Comparación con la inercia por bootstrap pareado en lugar de "intervalos que no se traslapan".
9. Criterio de holdout concreto.
10. La separación se mide en exceso sobre el azar (§8.2), no con $\eta^2$ crudo.
11. La prueba de independencia usa todos los desplazamientos, sin zona de exclusión (§8.3).
12. K-means con `scikit-learn` en lugar de `jumpmodels` con multa cero (§8.5).

### Cambios tras la revisión independiente del código

13. Nivel de azar por volteo de signos en bloques; el de desplazamiento queda como comparación que no decide (§8.2).
14. Amarre de cada etapa a la configuración, el snapshot y el commit registrados; reportes y resultado de holdout en la bitácora (§6).
15. Validación completa antes de abrir el holdout (§8.8).
16. Chequeo de cobertura de los datos (§3).
17. El ganador, y también el modelo más simple que propone FTIC, exigen estabilidad y separación positivas (§7); PBO con empates a medias (§8.6).
18. Hash de cada archivo de etiquetas en la bitácora; identidad del código por contenido; versiones de librerías amarradas (§6).

### Lo que los criterios pueden y no pueden detectar

Medido por el segundo revisor con datos sintéticos de 1,900 semanas:

| Tamaño real del efecto ($\eta^2$) | El criterio contra la inercia lo detecta |
|---|---|
| 0.01 | 15% de las veces |
| 0.02 | 40% |
| 0.05 | 94% |

Sin efecto real, ese criterio pasa entre 0.1% y 2.0% de las veces (nominal 2.5%) y la prueba de independencia entre 0.2% y 1.5% (nominal 1%). Elegir al mejor entre 24 configuraciones sin información sube la tasa de pasar ambos a un máximo de 3.7%.

Consecuencia: un efecto modesto pero real dará no-go la mayoría de las veces. Un no-go significa "no se distingue con claridad de la inercia", no "no hay fases".

El criterio de holdout, con unas 100 semanas, pasa cerca de la mitad de las veces sin efecto real y 63% con un efecto de 0.05. Por eso lleva a revisión y no bloquea.

### Límites que el código no cubre

- La regla "el holdout se abre una vez" vale por bitácora. Una bitácora nueva no recuerda la apertura anterior; eso es disciplina de proceso.
- Los archivos ignorados por git dentro de `src/` (por ejemplo `.pyc`) no cuentan en la identidad del código.

### Pendiente fuera de este spec

- Corregir en `docs/design/` los datos de series que resultaron equivocados (§3).
- Corregir la nota del vault sobre FTIC: es "Fan-Tang information criterion", no "Focused TIC".
- Costo del CJM con $K \ge 4$: la rejilla de probabilidades de 5% tiene 1,771 puntos con $K = 4$ y 10,626 con $K = 5$. El spec de confianza tendrá que usar pasos de 10% o limitar $K$.
