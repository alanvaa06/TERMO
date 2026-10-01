---
Writer: Claude
---

# TERMO — Diseño técnico

**Treasury Environment & Regime MOnitor.** Especificación técnica del modelo. Complementa al documento para comité, [`TERMO-diseno-comite.md`](TERMO-diseno-comite.md), que explica el *qué* y el *para qué*; aquí va el *cómo*. Solo diseño: no hay código; la construcción va en este repositorio.

**Convenciones del documento:**

| Marca | Significado |
|---|---|
| *(fuente)* | Respaldado por una fuente de la base de conocimiento del autor, citada en cursiva |
| *(diseño)* | Decisión propia de este diseño, sin fuente que la pruebe; se valida en §8 |
| *(por verificar)* | Dato citado de memoria, pendiente de confirmar antes de construir |

---

## 0. Objetivo técnico y alcance

**Objetivo.** Dada la información de la curva del Tesoro hasta el cierre de la semana $t$, producir:

| Salida | Definición |
|---|---|
| $\hat s_t$ | Etiqueta de régimen, $\hat s_t \in \{1,\dots,K\}$ |
| $\mathbf p_t$ | Probabilidades de régimen, $\mathbf p_t \in \Delta^{K-1}$ |
| $\boldsymbol\phi_t$ | Atribución por variable |
| $\mathcal C_t$ | Contexto histórico condicional: duración, transiciones y distribución del cambio futuro de la 10Y dado el régimen |

**Fuera de alcance:**
- Pronóstico puntual de tasas.
- Señales de trading.
- Optimización de portafolio.

TERMO es **contemporáneo**: etiqueta el presente. Su valor depende de que los regímenes sean persistentes y separen bien el comportamiento futuro, y ambas cosas se prueban en §8.

**Criterio de éxito:** pasar las pruebas de estabilidad y separación de §8. **No se optimiza Sharpe**, porque no hay estrategia.

---

## 1. Arquitectura

```
                      +--------------------+
FRED (diario) ------> | M1 Ingesta y QA    |  snapshot versionado
                      +---------+----------+
                                |
                      +---------v----------+
                      | M2 Features        |  PCA (L,S,C) + velocidades + vol
                      |   (diario -> sem.) |  z-score con stats de entrenamiento
                      +---------+----------+
                                |
              +-----------------+------------------+
              |                                    |
    +---------v----------+               +---------v----------+
    | M3 Motor de régimen|               | M6 Panel macro     |  fuera del modelo
    |  JM / CJM / SJM    |               |  ACM, futuros, CP  |
    +---------+----------+               +---------+----------+
              |  s_t, p_t, centroides              |
    +---------v----------+                         |
    | M4 Trazabilidad    |  surrogate XGB + SHAP   |
    |  + atribución exacta por centroides          |
    +---------+----------+                         |
              |                                    |
    +---------v------------------------------------v---------+
    | M5 Contexto histórico (duraciones, transiciones, fwd)  |
    +---------+----------------------------------------------+
              |
    +---------v----------+     +--------------------+
    | M7 Reporte/alertas | <-- | M8 Validación y    |  walk-forward, ARI,
    |  ficha mensual     |     |  bitácora de trials|  separación, PBO
    +--------------------+     +--------------------+
```

---

## 2. M1 — Datos

### 2.1 Series

| Uso | Serie (FRED) | Frecuencia |
|---|---|---|
| Curva | `DGS3MO`, `DGS1`, `DGS2`, `DGS3`, `DGS5`, `DGS7`, `DGS10`, `DGS20`, `DGS30` *(IDs por verificar)* | Diaria, días hábiles |
| Panel macro: prima por plazo | Términos de prima ACM, Fed de Nueva York *(por verificar: formato y frecuencia)* | Diaria |
| Panel macro: expectativas | Futuros de fed funds (≤6m) y SOFR a 3 meses (más allá), CME | Diaria |
| Panel macro: Cochrane-Piazzesi | Forwards a 1–5 años construidos de la curva o del dataset Fama-Bliss | Mensual / diaria |

La elección de futuros por horizonte viene de *Futures-Implied Policy Expectations and Bond Risk Premia* *(fuente)*.

### 2.2 Historia y huecos

*(por verificar)*

- `DGS2` empezaría en 1976.
- `DGS30` tendría un hueco entre 2002 y 2006, cuando no hubo emisión del bono a 30 años.
- `DGS20` tendría su propio hueco.

| Opción | Muestra | Costo |
|---|---|---|
| A | Desde ~1977, curva completa salvo 20/30 en los huecos | Menos historia |
| B | Desde 1962 sin el 2Y | Pierde el tramo corto, que importa para *bear flatteners* |

*(diseño)* **Recomendación: A.** El PCA de §3.1 se ajusta solo con plazos sin huecos (3M, 1, 2, 3, 5, 7, 10Y). El 20Y y el 30Y entran solo en la pendiente 10s30s; en el hueco esa variable se marca faltante y el motor la ignora (§4.5).

### 2.3 Calidad y versionado *(diseño)*

- **Calendario:** días hábiles de EE.UU. No se interpola: un faltante aislado se arrastra máximo 1 día y si hay más se marca.
- **Snapshot inmutable por corrida:** hash del archivo de datos y fecha de descarga. Las series CMT de FRED rara vez se revisan, pero el snapshot hace reproducible cada lectura histórica *(por verificar: política de revisiones)*.
- **Nota técnica:** las series `DGS*` son tasas CMT *par*, no cupón cero. Para el factor de Cochrane-Piazzesi hacen falta forwards cero. Opciones:
  - Bootstrapping de la curva par.
  - Usar el dataset Fama-Bliss publicado por Cochrane, que es la opción de *Futures-Implied Policy Expectations and Bond Risk Premia* *(fuente)*.

---

## 3. M2 — Features

Muestreo: se calculan diario y se leen al **cierre del último día hábil de cada semana** *(diseño)*.

### 3.1 Forma de la curva: PCA

Sea $\mathbf y_t \in \mathbb R^{m}$ el vector de tasas en los $m$ plazos sin huecos.

- **PCA sobre cambios diarios** $\Delta\mathbf y_t$, ajustado **solo con la ventana de entrenamiento** y congelado hasta el siguiente reajuste:
$$\mathbf p_1 = \arg\max_{\lVert\mathbf w\rVert=1}\ \mathbf w^\top \Sigma_{\Delta y}\,\mathbf w, \quad \text{y así sucesivamente tras deflactar}$$
- En el ejemplo de Dixon, los 3 primeros componentes explican **95.6% / 4.07% / 0.34%** y se leen como **nivel, pendiente y curvatura** (*ML in Finance (Dixon et al.) — Part II Sequential* *(fuente)*).
- **Puntajes en nivel:** $L_t, S_t, C_t = \mathbf p_{1,2,3}^\top(\mathbf y_t - \bar{\mathbf y})$.

**Convención de signo** *(diseño)*. Fijar el signo de cada componente para que sea legible en el reporte:

| Componente | Convención |
|---|---|
| $L$ | Positivo cuando sube la 10Y |
| $S$ | Positivo cuando la curva se empina |
| $C$ | Positivo cuando la parte media sube respecto a los extremos |

Sin esta convención, un reajuste puede invertir el signo y romper la comparación histórica.

### 3.2 Velocidad con suavizado multi-ventana

Cambio de un puntaje $X \in \{L,S,C\}$ a horizonte $h$ días hábiles:
$$\Delta_h X_t = X_t - X_{t-h}$$

**Suavizado de efecto base** *(diseño, técnica conocida)*. Promediar ventanas vecinas que comparten el mismo extremo $t$:
$$\widetilde\Delta_h X_t = \tfrac13\big(\Delta_{h-\delta}X_t + \Delta_h X_t + \Delta_{h+\delta}X_t\big)$$

- Diluye el efecto de que un movimiento viejo **salga** de la ventana, sin amortiguar lo que pasa hoy.
- Horizontes: $h \in \{21, 63, 126\}$ con $\delta \in \{5, 10, 15\}$ *(diseño, a calibrar)*.

### 3.3 Volatilidad y aceleración

Volatilidad EWM de cambios diarios de tasa, para los plazos 2Y, 5Y y 10Y, con vidas medias $hl \in \{20, 60, 120\}$:
$$\sigma^{(hl)}_{j,t} = \sqrt{\textstyle\sum_{k\ge0} w_k^{(hl)}\,(\Delta y_{j,t-k})^2}, \qquad w_k \propto 2^{-k/hl}$$

Se usan **diferencias entre vidas medias** en lugar de niveles redundantes:
$$\sigma^{(20)} - \sigma^{(60)}, \qquad \sigma^{(60)} - \sigma^{(120)}$$

- En acciones, las diferencias entre vidas medias quitaron colinealidad y metieron una señal de impulso (3 vol + 1 retorno) (*Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)* *(fuente)*).
- Allí se prefirieron vidas medias largas: con vidas medias cortas, los regímenes no persisten ni con multa alta.

**Variante asimétrica** *(diseño, hipótesis)*. Semidesviación solo de los movimientos **al alza** de tasa: $\sigma^{+}$ usa $\max(\Delta y,0)$.
- Es el análogo de la *downside deviation*, que en acciones evitó etiquetar como bajista un rally volátil.
- **Sin evidencia en tasas dentro de la base de conocimiento.** Entra como candidata en la bitácora de §8.4, no por defecto.

### 3.4 Vector final y estandarización

| Bloque | Variables | Dim. |
|---|---|---|
| Forma | $L_t, S_t, C_t$ | 3 |
| Velocidad | $\widetilde\Delta_{21}, \widetilde\Delta_{63}$ de $L, S$; $\widetilde\Delta_{63}$ de $C$ | 5 |
| Nervio | $\sigma^{(60)}$ en 5Y | 1 |
| Aceleración | $\sigma^{(20)}-\sigma^{(60)}$ y $\sigma^{(60)}-\sigma^{(120)}$ en 5Y; $\sigma^{(20)}-\sigma^{(60)}$ en 2Y y 10Y | 4 |
| Pendiente larga | 10s30s (nivel) | 1 |
| **Total base** | | **~14** |

- **Estandarización:** z-score con media y desviación **de la ventana de entrenamiento**, congeladas hasta el reajuste.
- **Por qué importa:** el agrupamiento pondera cada variable por igual en $\ell_2$, así que la dimensión y la colinealidad pesan más que en una regresión (*Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)* *(fuente)*).
- **Control de colinealidad** *(diseño)*: si $|\rho| > 0.8$ entre dos variables en entrenamiento, se elimina una antes de agrupar. La decisión se registra en la bitácora.

---

## 4. M3 — Motor de régimen

### 4.1 Jump model estadístico (JM)

Dadas variables estandarizadas $\mathbf x_t \in \mathbb R^D$, se estiman centroides $\Theta = \{\boldsymbol\mu_k\}$ y la secuencia de estados $S = (s_0,\dots,s_{T-1})$ (*Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)* *(fuente)*):
$$\min_{\Theta, S}\ \sum_{t=0}^{T-1} \tfrac12\lVert \mathbf x_t - \boldsymbol\mu_{s_t}\rVert_2^2 \;+\; \lambda \sum_{t=1}^{T-1} \mathbb 1\{s_{t-1}\neq s_t\}$$

| $\lambda$ | Comportamiento |
|---|---|
| $0$ | Exactamente K-means: ignora el orden temporal y parpadea |
| Moderado | Regímenes persistentes |
| $\to\infty$ | Colapsa a un solo estado |

**Estimación** *(fuente)*:
- Descenso coordenado: con $S$ fijo, $\boldsymbol\mu_k$ es la media del cluster; con $\Theta$ fijo, $S$ sale por programación dinámica tipo Viterbi sobre costo más multa.
- 10 reinicios desde k-means++ por la no convexidad.

**Inferencia en línea** *(fuente)*. Con parámetros congelados se minimiza solo sobre la secuencia y se reporta el último estado. Es la primitiva de producción: `.predict_online()` en *jumpmodels (Python Library)*.

Evidencia en acciones, prueba 1990–2023 con 10 pb de costo *(fuente)*:

| | Cambios de régimen | Rotación |
|---|---|---|
| JM | 14 | ~5x menor |
| HMM | 115 | — |

### 4.2 Jump model continuo (CJM): probabilidades nativas

- El estado pasa a ser un **vector de probabilidad** $\mathbf p_t \in \Delta^{K-1}$ en lugar de una etiqueta dura (Aydınhan, Kolm, Mulvey, Shu, 2024) *(fuente: *jumpmodels (Python Library)*)*.
- Implementado en la misma clase `JumpModel`, con `.predict_proba_online()`.
- **La formulación exacta del CJM no está compilada en la base de conocimiento.** Hay que leer el paper antes de fijar hiperparámetros propios del CJM.

**Por qué CJM y no un clasificador encima.** La probabilidad sale del mismo objetivo que define los regímenes, sin un segundo modelo que imite etiquetas. Eso elimina la circularidad de la confianza.

### 4.3 Jump model disperso (SJM): selección de variables

- Agrega pesos por variable con penalización $\ell_1$ ($\kappa$), de modo que solo un subconjunto de variables define los regímenes (Nystrup, Kolm, Lindström, 2021) *(fuente: *Generalized Information Criteria for Jump Models (Cortese 2026)*)*.
- En acciones, con 125 variables candidatas, se seleccionaron 55, y dominaron las EMAs de volatilidad *(fuente)*.
- *(diseño)* Con ~14 variables base el SJM es **opcional**: se corre como diagnóstico de qué variables importan y se adopta solo si mejora estabilidad y separación en §8.

### 4.4 Selección de $K$, $\lambda$, $\kappa$

**Etapa 1: ajuste estadístico con FTIC** *(fuente: *Generalized Information Criteria for Jump Models (Cortese 2026)*)*

$$\text{IC} = -2\,\log \hat L \;+\; \text{penalty}\big(\text{complejidad efectiva del SJM}\big)$$

- La complejidad usa una aproximación de primer orden.
- En simulaciones, el **FTIC** identifica $K$ correcto con ARI = 1.00 y es conservador con variables falsas. AIC y BIC fallan en $K=2$.
- Persistencia típica en mercados ≈ 87.5%, que es el escenario relevante.
- En el empírico MSCI/MSCIEM eligió $\lambda=10$, $\kappa=6$, $K=3$, con la volatilidad como motor dominante.

**Etapa 2: validación por objetivo del producto** *(diseño)*

En acciones, $\lambda$ se eligió maximizando el Sharpe de validación *(fuente)*. TERMO no tiene estrategia, así que se sustituye por un **puntaje compuesto de contexto**:
$$\mathcal J(K,\lambda,\kappa) = \underbrace{\overline{\text{ARI}}_{\text{estab}}}_{\S 8.2} \times \underbrace{\eta^2_{\text{sep}}}_{\S 8.3}$$
con restricción de duración mínima mediana de régimen $\ge 4$ semanas.

**Grilla** *(diseño)*:

| Hiperparámetro | Valores |
|---|---|
| $K$ | $\{2,3,4,5\}$ |
| $\lambda$ | 8–10 valores log-espaciados |
| $\kappa$ | 4–5 valores, si se usa el SJM |

Total ≈ 150–200 configuraciones. **Todas van a la bitácora (§8.4).** Con este $N$ aplica la advertencia de MinBTL: con 5 años de datos diarios, más de ~45 configuraciones independientes casi garantizan un "ganador" espurio *(fuente: *Deflated Sharpe Ratio and Backtest Overfitting*)*. Por eso la grilla se poda por teoría antes de correrla.

**Regla de desempate** *(diseño)*: si FTIC y $\mathcal J$ discrepan en $K$, gana el **$K$ menor** que pase los criterios de §8.6. Un modelo más simple es más defendible.

### 4.5 Faltantes

*(diseño)* Si en una semana falta una variable (por ejemplo 10s30s en el hueco del 30Y), la distancia al centroide se calcula solo sobre las variables observadas y se reescala por $D/D_{obs}$. Se marca la lectura como "parcial".

### 4.6 Nombres estables entre reajustes (*label switching*)

*(diseño)* Cada reajuste puede permutar los índices de cluster. Para que "venta extrema" siga siendo "venta extrema":

1. **Emparejar centroides nuevos con los anteriores** con el algoritmo húngaro sobre la matriz de distancias $\lVert\boldsymbol\mu^{new}_i - \boldsymbol\mu^{old}_j\rVert$.
2. **Asignar nombres con reglas sobre el centroide** (una regla por régimen hipotético):

| Régimen | Regla |
|---|---|
| Venta extrema | $\widetilde\Delta_{21} L \gg 0$, $\widetilde\Delta_{21} S < 0$, aceleración de vol $> 0$ |
| Rally extremo | $\widetilde\Delta_{21} L \ll 0$, $\widetilde\Delta_{21} S > 0$ |
| Venta moderada / rally moderado | Signo de $\widetilde\Delta_{63} L$ con vol bajo la mediana |

3. **Bloquear el reajuste:** si el emparejamiento deja un centroide con distancia mayor al umbral, se marca "régimen nuevo o redefinido" y requiere revisión humana antes de publicar.

---

## 5. M4 — Confianza y trazabilidad

### 5.1 Confianza

- **Probabilidades:** $\mathbf p_t$ del CJM.
- **Incertidumbre normalizada:**
$$H_t = -\frac{1}{\log K}\sum_k p_{t,k}\log p_{t,k} \in [0,1]$$
- **Margen:** $m_t = p_{t,(1)} - p_{t,(2)}$, la diferencia entre la primera y la segunda opción.
- **Bandas de lectura** *(diseño, umbrales a calibrar en §8)*:

| Banda | Condición |
|---|---|
| Clara | $p_{(1)} > 0.7$ |
| Probable | $0.5$–$0.7$ |
| Posible transición | $p_{(1)} < 0.5$ o $m_t < 0.15$ |

- **Calibración:** confiabilidad de $\mathbf p_t$ contra frecuencias reales de etiqueta en walk-forward. Si está mal calibrada, isotónica por régimen.

### 5.2 Capa A — surrogate XGBoost + SHAP

1. **Surrogate:** XGBoost multiclase entrenado sobre las mismas variables $\mathbf x_t$, con objetivo $\hat s_t$ del JM en la ventana de entrenamiento.
   - *(diseño)* Profundidad acotada (3–4) y un mínimo de muestras por hoja.
   - Los árboles a profundidad completa sobreajustan; hay que acotar profundidad y tamaño de hoja.
2. **Fidelidad** *(diseño)*: concordancia surrogate–JM fuera de muestra (walk-forward).
   - **Si la fidelidad es menor a 90%, el SHAP no se publica**: explicaría a otro modelo.
3. **SHAP** (Lundberg & Lee, 2017), TreeExplainer. Para cada clase $k$, descomposición aditiva en log-odds:
$$f_k(\mathbf x_t) = \phi^{(k)}_0 + \sum_{j=1}^{D}\phi^{(k)}_{j}(\mathbf x_t), \qquad p^{sur}_{t,k} = \operatorname{softmax}_k\big(f(\mathbf x_t)\big)$$
   - Se reporta $\phi^{(\hat s_t)}_j$, es decir, qué empuja hacia el régimen elegido.
   - **Para el comité** se traduce aproximando el efecto marginal en probabilidad. El documento de comité lo muestra como "ticket" ilustrativo; la suma exacta es en log-odds.
   - **Ventaja sobre las importancias clásicas:** explica predicciones individuales, agrega de forma consistente y separa por clase. Las importancias clásicas (gain, cover) pueden contradecirse entre sí.
4. **Restricciones monótonas** *(diseño, opcional)*. XGBoost permite forzar el signo del efecto de una variable. Candidata: $\partial f_{\text{venta}}/\partial\, \widetilde\Delta_{21} L \ge 0$. Solo donde haya prior fuerte, y se registra en la bitácora.

### 5.3 Capa B — atribución exacta por centroides (sin surrogate)

La pérdida del JM es separable en $\ell_2$, así que la preferencia del régimen elegido $k^*$ frente al segundo $k_2$ se descompone **exactamente** por variable *(diseño; álgebra directa del objetivo de §4.1)*:
$$\Delta_j(\mathbf x_t) = \tfrac12\big(x_{t,j}-\mu_{k_2,j}\big)^2 - \tfrac12\big(x_{t,j}-\mu_{k^*,j}\big)^2, \qquad \sum_j \Delta_j = d_{k_2}(\mathbf x_t) - d_{k^*}(\mathbf x_t)$$

- $\Delta_j > 0$: la variable $j$ acerca el día al régimen elegido.
- **No incluye la multa por salto:** la inercia de régimen se reporta aparte como "persistencia".

### 5.4 Regla de consistencia A/B *(diseño)*

1. Correlación de Spearman entre el ranking de $\lvert\phi^{(\hat s_t)}_j\rvert$ y el de $\Delta_j$ en las variables top-5.
2. Si $\rho_S < 0.5$ o las variables top-2 no coinciden en signo, la lectura se marca **"a revisar"**.

### 5.5 Historial de impulsores

Mapa de calor semanal de $\Delta_j$ (o $\phi_j$) por variable, de 52 semanas, para detectar variables que **cambian de papel**: pasar de frenar a empujar.

---

## 6. M5 — Contexto histórico condicional

Todo se calcula en walk-forward, usando solo información disponible a la fecha de cada etiqueta.

| Estadístico | Definición | Nota |
|---|---|---|
| Duración (sojourn) | Distribución de semanas consecutivas por régimen | Mediana y p25–p75 |
| Matriz de transición | $\hat P_{ij}(h) = \Pr(s_{t+h}=j \mid s_t=i)$, $h \in \{4,13,26\}$ semanas | IC por *block bootstrap* |
| Cambio futuro de la 10Y | $\Delta y^{10}_{t\to t+h}$ por régimen, $h \in \{4, 13\}$ semanas | **Todas** las semanas en régimen, no solo inicios de episodio; esto evita el sesgo de supervivencia |
| Duración restante condicional | $\Pr(\text{sigue en } i \text{ a } h \mid \text{lleva } d \text{ semanas en } i)$ | Base del "qué suele venir después" |

**Por qué *block bootstrap*:** con horizontes solapados las observaciones están autocorrelacionadas. El bootstrap simple subestima los intervalos. Bloques de longitud ≥ $h$ *(diseño)*.

---

## 7. M6 — Panel macro (no entra al agrupamiento)

| Indicador | Construcción | Lectura |
|---|---|---|
| Prima por plazo 10Y | Estimación ACM de la Fed de NY *(por verificar acceso)*; nivel y cambio a 4/13 semanas | Si la 10Y sube por riesgo o por expectativas |
| Trayectoria implícita de política | Futuros de fed funds hasta 6m, SOFR a 3 meses después | Subidas o recortes esperados |
| Factor Cochrane-Piazzesi | Ver abajo | Prima de riesgo esperada en bonos a 1 año |

**Factor Cochrane-Piazzesi** *(fuente: *Futures-Implied Policy Expectations and Bond Risk Premia*)*:
$$rx^{(n)}_{t+1} = b_n\big(\gamma_0 + \gamma_1 y^{(1)}_t + \gamma_2 f^{(2)}_t + \dots + \gamma_5 f^{(5)}_t\big) + \varepsilon, \qquad CP_t = \boldsymbol\gamma^\top \mathbf f_t$$

- Una sola combinación "en tienda" de forwards predice excesos de rendimiento de bonos a 1–5 años con $R^2$ de hasta 0.44.
- Es contracíclica y contiene información que no está en nivel, pendiente y curvatura.
- *(diseño)* $\boldsymbol\gamma$ se estima en la ventana de entrenamiento y se congela, igual que todo lo demás.

**Advertencias** *(fuente)*:
- La prima en futuros de fed funds (41–73 pb/año) es un promedio 1988–2003 y **puede cambiar de signo** (Diercks-Carl).
- En los horizontes relevantes, la prima domina a la convexidad por un orden de magnitud.
- El panel se presenta como **niveles y z-scores**, nunca como "señal".

---

## 8. M8 — Protocolo de validación

### 8.1 Esquema temporal

| Elemento | Especificación |
|---|---|
| Esquema | **Walk-forward** con reajuste cada 26 semanas; cada reajuste usa solo el pasado (*ML in Finance (Dixon et al.) — Part II Sequential* *(fuente)*) |
| Ventana | *(decisión abierta)* **Expansiva** (más datos, menos control del tamaño de muestra) vs **móvil** (en acciones se usaron ~2000 días) |
| Purga / embargo | Entre entrenamiento y evaluación, embargo ≥ horizonte máximo de las variables (126 días) para que ninguna ventana de cambio cruce el corte *(diseño)* |
| Holdout reservado | Últimas ~104 semanas (2024–2026): **una sola evaluación**, al final, después de congelar todo |

**Punto pendiente.** La teoría completa de purga y embargo (validación cruzada purgada) no está compilada en la base de conocimiento. Solo existe el índice en *Advances in Financial ML (López de Prado) — Book Map*, y ninguna de las variantes de CV de Kaabar purga ni embarga (*Deep Learning for Finance (Kaabar 2024) — Advanced Techniques*).

### 8.2 Prueba 1 — Estabilidad

**ARI** entre etiquetados del mismo periodo producidos por modelos entrenados en submuestras distintas *(métrica usada en *Generalized Information Criteria for Jump Models (Cortese 2026)*)*:
$$\text{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]\big/\binom{n}{2}}{\tfrac12\big[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\big] - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]\big/\binom{n}{2}}$$

**Diseño** *(diseño)*:
- (a) Mitades de la muestra.
- (b) Reajustes consecutivos sobre su periodo común.
- (c) Bootstrap por bloques de años.

Se reporta $\overline{\text{ARI}}$ y su mínimo.

### 8.3 Prueba 2 — Separación

Para $h \in \{4, 13\}$ semanas, con $\Delta y^{10}_{t\to t+h}$ agrupado por régimen:
- **Efecto:** $\eta^2 = SS_{\text{entre}} / SS_{\text{total}}$.
- **Contraste no paramétrico:** Kruskal-Wallis.
- **Intervalos:** *block bootstrap* para medianas y p10–p90 por régimen.
- **Independencia:** tabla de contingencia régimen × signo del cambio futuro contra independencia con $\chi^2$. Un clasificador con 58% de acierto puede ser indistinguible del ruido; hay que probar siempre contra independencia (*ML in Finance (Dixon et al.) — Part II Sequential* *(fuente)*).

### 8.4 Prueba 3 — Contra lo obvio

| Línea base | Qué captura |
|---|---|
| Incondicional | Distribución de $\Delta y^{10}$ sin régimen |
| Signo de $\widetilde\Delta_{63}L$ | Inercia simple |
| K-means ($\lambda=0$) | Mide cuánto aporta la multa |
| HMM gaussiano, mismo $K$ | Alternativa clásica |

TERMO debe superar a las cuatro en $\eta^2$ y en estabilidad. Si no le gana a la inercia simple, **no aporta contexto nuevo**.

### 8.5 Prueba 4 — Honestidad: bitácora, $N$ efectivo y PBO

**Bitácora**, adaptada del protocolo de *Deflated Sharpe Ratio and Backtest Overfitting* *(fuente, adaptado)*. Se registra **antes** de mostrar el resultado:

| Campo | Contenido |
|---|---|
| `trial_id`, hipótesis | Qué se prueba y por qué |
| Configuración | $K,\lambda,\kappa$, variables, ventana, $\delta$ |
| Métricas | ARI, $\eta^2$, duración mediana, fidelidad del surrogate |
| Serie de etiquetas | Para correlación entre trials |
| Estado | Mantenido / descartado / falló, con motivo |

**$N$ efectivo.**
- Agrupar las series de etiquetas de los trials (distancia $1-\text{ARI}$).
- El número de clusters es $N_{\text{ef}}$.
- Reportar $M$ bruto y $N_{\text{ef}}$.

**PBO por CSCV** *(adaptación de diseño)*. PBO no depende de que la métrica sea Sharpe: es la probabilidad de que la configuración ganadora dentro de muestra quede bajo la mediana fuera de muestra *(fuente)*.
1. Matriz $T\times N$ de contribuciones semanales a $\eta^2$ por trial.
2. $S=16$ bloques temporales, lo que da $\binom{16}{8}=12{,}870$ combinaciones.
3. Rango dentro y fuera de muestra, logit $\lambda_c$.
4. $\widehat{PBO} = \Pr(\lambda_c<0)$.
5. Rechazo sugerido si $\widehat{PBO}>0.05$ *(fuente)*.

PBO complementa al walk-forward, no lo reemplaza: recombinar bloques rompe el orden temporal en las fronteras.

### 8.6 Pruebas de hipótesis de dominio

| Hipótesis | Prueba |
|---|---|
| Regímenes extremos cortos, moderados largos | Distribución de duraciones por régimen (§6) |
| Una venta rara vez pasa directo a rally | $\hat P_{\text{venta}\to\text{rally}}(4)$ con IC; debe ser menor que la probabilidad incondicional de rally |
| Un régimen no mezcla episodios distintos (por ejemplo, un rally calmado y uno de crisis) | Dispersión intra-régimen de $\sigma^{(60)}$; prueba de bimodalidad (dip test); si es bimodal, probar $K+1$ |

### 8.7 Criterios de aceptación

*(diseño; umbrales propuestos, a ratificar por el comité)*

| Criterio | Umbral propuesto | Si falla |
|---|---|---|
| $\overline{\text{ARI}}$ de estabilidad | ≥ 0.6 | No se presenta |
| $\eta^2$ a 4 semanas | > línea base de inercia, con IC que no se traslapa | No se presenta |
| $\chi^2$ régimen × signo | $p < 0.01$ | No se presenta |
| Duración mediana | ≥ 4 semanas en todos los regímenes | Subir $\lambda$ o reducir $K$ |
| Fidelidad del surrogate | ≥ 90% | Se publica sin SHAP (solo capa B) |
| PBO | ≤ 0.05 | Podar la grilla y repetir |
| Holdout final | Sin degradación material frente al walk-forward | Revisión antes de producción |

---

## 9. M7 — Operación

### 9.1 Calendario

| Reloj | Frecuencia | Proceso |
|---|---|---|
| Ingesta | Diaria | Descarga, QA (§2.3), snapshot |
| **Lectura** | **Semanal** (cierre del último día hábil) | Variables, inferencia en línea con parámetros congelados, $\mathbf p_t$, capas A/B, consistencia |
| **Alerta** | En la lectura semanal | Ver la regla de §9.2 |
| **Reporte** | **Mensual** | Ficha (ver documento de comité) + historial de impulsores |
| **Reajuste** | **Semestral** | Ver el proceso de §9.3 |

### 9.2 Regla de alerta

*(diseño)* Se envía alerta si $\hat s_t \neq \hat s_{t-1}$ **y** $p_{t,\hat s_t} \ge 0.6$ **y** la consistencia A/B no está marcada "a revisar".

Si solo se cumple el cambio de etiqueta, se registra "transición en curso" sin alerta. El JM ya es persistente; la condición de confianza evita alertar por empates.

### 9.3 Proceso de reajuste semestral

1. Re-estimar PCA, z-scores, $\boldsymbol\gamma$ de CP, JM/CJM y el surrogate con la ventana nueva.
2. Emparejar regímenes (§4.6).
3. Comparar el etiquetado histórico viejo contra el nuevo en el periodo común (ARI).
   - Si ARI < 0.6, **se congela la versión anterior** y se revisa.
4. **Shadow:** 4 semanas de lecturas paralelas viejo/nuevo. Se publica la vieja y se registran las diferencias.
5. Promoción con versión nueva.

El esquema evaluación → shadow → despliegue con vuelta atrás probada se adapta de *Operationalizing AI Agents in Finance (Packt Ch12)* *(fuente)*. Allí, un cambio de una línea movió 18% de los veredictos y solo el modo shadow lo detectó.

**Por qué semestral:** reentrenar con más frecuencia apenas mejoró en Kaabar (de 48.55% a 48.92% de acierto) y cambia el significado de los regímenes (*Deep Learning for Finance (Kaabar 2024) — Advanced Techniques* *(fuente)*).

### 9.4 Versionado y auditoría

- **Manifiesto de versión:** hash de {snapshot de datos, configuración, $K,\lambda,\kappa$, cargas del PCA, centroides, surrogate, $\boldsymbol\gamma$ de CP, commit del código}.
  - El ID de versión es el hash del JSON canónico *(fuente: *Operationalizing AI Agents in Finance (Packt Ch12)*, adaptado)*.
- **Registro por lectura:** fecha, versión, $\mathbf x_t$, $\mathbf p_t$, $\hat s_t$, $\boldsymbol\phi_t$, $\boldsymbol\Delta_t$, bandera de consistencia, panel macro, alerta sí/no.

### 9.5 Monitoreo de deriva

*(fuente para los umbrales: *Evaluating Financial Agents — KYC Case Study (Packt Ch11)*, adaptado a variables de mercado)*
- **PSI y KS** de cada variable de la ventana reciente (13 semanas) contra la distribución de entrenamiento.
  - PSI aviso 0.10, alerta 0.25.
- **Deriva de distancia mínima a centroides:** $\min_k d_k(\mathbf x_t)$ por encima del p99 de entrenamiento indica que **hoy no se parece a ningún régimen conocido**. Se reporta como "fuera de experiencia" en la ficha.

---

## 10. Estructura propuesta del repositorio (diseño)

```
TERMO/
├── docs/design/          # diseño comité + este diseño técnico
├── configs/              # grillas, horizontes, umbrales (versionados)
├── src/termo/
│   ├── data/             # M1 ingesta FRED, QA, snapshots
│   ├── features/         # M2 PCA, velocidades, vol, z-score
│   ├── regime/           # M3 JM/CJM/SJM, selección, label matching
│   ├── explain/          # M4 surrogate+SHAP, atribución por centroides
│   ├── context/          # M5 duraciones, transiciones, forward stats
│   ├── macro/            # M6 ACM, futuros, Cochrane-Piazzesi
│   ├── validation/       # M8 walk-forward, ARI, η², PBO, bitácora
│   └── report/           # M7 ficha mensual, alertas
├── trials/               # bitácora inmutable (append-only)
└── tests/
```

---

## 11. Riesgos técnicos

| Riesgo | Mitigación |
|---|---|
| Pocos episodios independientes en ~50 años | Grilla podada por teoría; $N_{\text{ef}}$ y PBO; preferir $K$ menor |
| La evidencia del jump model es de acciones, no de tasas | Las pruebas 1–3 son el criterio, no la literatura |
| Reacción tardía con $\lambda$ alto: en acciones no detectó el crash de 1987 (*Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)*) | Restricción de duración mínima, no máxima; la banda "posible transición" avisa antes del cambio de etiqueta |
| Cambio de signo o rotación del PCA entre reajustes | Convención de signo (§3.1) y emparejamiento de centroides (§4.6) |
| El surrogate explica algo distinto al JM | Umbral de fidelidad y capa B exacta |
| Régimen inédito | Deriva de distancia mínima, bandera "fuera de experiencia" |
| SHAP leído como causalidad | Advertencia fija en la ficha; el panel macro es el canal causal |
| Disponibilidad o formato de ACM y futuros | Verificación previa (§2); el panel es opcional para la lectura del régimen |

---

## 12. Decisiones técnicas abiertas

1. **Ventana expansiva vs móvil** (§8.1).
2. **Base del PCA:** ¿cambios diarios (como en el ejemplo de Dixon) o niveles? *(diseño: cambios, por estacionariedad)*
3. **SJM sí o no** con ~14 variables (§4.3).
4. **Variante asimétrica de vol** (§3.3): ¿entra a la grilla?
5. **Umbrales de §8.7**: ratificar con el comité.
6. **Muestra desde 1977 (opción A) vs desde 1962 sin el 2Y (opción B)** (§2.2).
7. **CJM:** leer Aydınhan et al. (2024) antes de fijar sus hiperparámetros, porque la formulación no está en la base de conocimiento.

---

## 13. Referencias

**Base de conocimiento:**
- *Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)*: objetivo del JM, estimación, selección por validación, variables de diferencias de vol.
- *Generalized Information Criteria for Jump Models (Cortese 2026)*: SJM, FTIC, ARI.
- *jumpmodels (Python Library)*: JM/CJM/SJM, inferencia en línea; referencias a Nystrup 2020/2021 y Aydınhan 2024.
- *ML in Finance (Dixon et al.) — Part II Sequential*: PCA de curva, walk-forward, prueba contra independencia, HMM.
- *Deflated Sharpe Ratio and Backtest Overfitting*: bitácora, $N$ efectivo, PBO/CSCV, MinBTL.
- *Futures-Implied Policy Expectations and Bond Risk Premia*: Cochrane-Piazzesi, prima en futuros, elección de contrato.
- *Deep Learning for Finance (Kaabar 2024) — Advanced Techniques*: reentrenamiento, CV móvil/expansiva sin purga.
- *Evaluating Financial Agents — KYC Case Study (Packt Ch11)*: umbrales PSI/KS de deriva.
- *Operationalizing AI Agents in Finance (Packt Ch12)*: manifiesto de versión, shadow y canary.

**Pendientes de compilar** (huecos que este diseño necesita):
- Validación cruzada purgada y embargo: López de Prado, *Advances in Financial ML* (solo índice en *Advances in Financial ML (López de Prado) — Book Map*).
- Formulación del CJM (Aydınhan, Kolm, Mulvey, Shu, 2024).
- Estimación de prima por plazo ACM (Adrian, Crump, Moench).
- SHAP: Lundberg & Lee (2017) como paper primario.
