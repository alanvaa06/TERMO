---
Writer: Claude
---

# TERMO — Treasury Environment & Regime MOnitor

**Termómetro de fases del mercado de bonos del Tesoro.**

**Documento de diseño para comité.** Solo diseño: no hay código. La construcción se hará en este repositorio.
Base: literatura académica sobre detección de regímenes (jump models), evaluación honesta de modelos y prima de riesgo en bonos, compilada en la base de conocimiento del autor.

---

## 1. Resumen en una página

**Qué es.** TERMO es una herramienta de **contexto** que cada semana dice:
- En qué **fase** está el mercado de bonos del Tesoro de EE.UU. (por ejemplo: "venta extrema" o "rally moderado").
- **Con qué confianza** lo dice.
- **Qué datos la empujan.**
- **Qué suele venir después**, según ~50 años de historia.

**Qué no es.** No predice la tasa. No da órdenes de compra o venta. Es un termómetro, no un piloto automático.

**Qué lo distingue del enfoque clásico** de agrupar días con K-means:
1. **Fases estables** sin necesidad de parches.
2. **Confianza real**, no heredada de un segundo modelo.
3. **Pruebas formales** de que las fases son reales antes de presentarlas.
4. **Trazabilidad completa:** cada lectura viene con su explicación.

**Cómo se usa.**

| Qué | Frecuencia |
|---|---|
| Lectura del modelo | Semanal, automática |
| Ficha al comité | Mensual |
| Alerta | Cuando cambia de fase con confianza alta |
| Reentrenamiento | Cada 6 meses |

**Datos.** Públicos y gratuitos: la curva del Tesoro de FRED. El panel macro usa fuentes públicas de la Fed y de CME.

---

## 2. El ecosistema: dónde encaja

TERMO es el primer módulo de una familia de herramientas de contexto. La idea es que el comité vea **varias fases a la vez**, cada una en su mercado:

| Módulo | Mercado | Pregunta | Tipo | Estado |
|---|---|---|---|---|
| **TERMO** | **Tesoro EE.UU.** | **¿En qué fase están los bonos?** | **Contexto (hoy)** | **Este documento** |
| Termómetro de bolsa | Bolsa global | ¿En qué fase está la bolsa? | Contexto (hoy) | Posible extensión |
| Termómetro de materias primas | Commodities | ¿En qué fase están? | Contexto (hoy) | Posible extensión |
| Termómetro de emergentes | Bolsas EM por país | ¿Qué tan alcista está cada país? | Contexto (hoy) | Posible extensión |
| Módulo de dirección | Tasa a 10 años | ¿Sube o baja el próximo mes? | Predicción (1 mes) | Segunda etapa |

**Cómo se relacionan:**
- **TERMO le da contexto a cualquier predicción.** Un modelo de dirección puede funcionar bien en fases tranquilas y mal en las extremas. Saber la fase dice cuánto creerle.
- **Bonos + bolsa juntos** responden si los dos mercados están en fases compatibles. Por ejemplo: venta de bonos con la bolsa subiendo con fuerza es un mundo de crecimiento, no de pánico.

---

## 3. Punto de partida: el enfoque clásico

La forma estándar de detectar fases en un mercado tiene tres piezas:

1. **K-means** agrupa días parecidos en un número fijo de fases.
2. **Un clasificador** (por ejemplo XGBoost) aprende a reconocer esas fases y da una probabilidad diaria.
3. **SHAP** explica qué datos empujaron la decisión.

**Hipótesis inicial de fases para bonos.** Son conceptos estándar del mercado de tasas; el paso 3 decide si son 4 o menos:

| Fase | Qué pasa con las tasas |
|---|---|
| Venta moderada | Suben poco a poco |
| Venta extrema (*bear flattener*) | Suben fuerte, sobre todo las de 2 a 5 años; la curva se aplana |
| Rally moderado | Bajan poco a poco |
| Rally extremo (*bull steepener*) | Bajan fuerte; la curva se empina |

**Hipótesis a confirmar en el paso 7:**
- Las fases extremas son cortas (semanas) y las moderadas largas.
- Una venta rara vez pasa directo a rally; primero se modera.

---

## 4. Las fallas del enfoque clásico que corregimos

| # | Falla | Por qué le importa al comité | Se corrige en |
|---|---|---|---|
| 1 | K-means no ve el tiempo: las fases parpadean | Una herramienta que cambia de opinión cada semana no sirve para decidir | Paso 3 |
| 2 | La confianza sale de un modelo que imita a otro | La confianza reportada no es confianza real | Paso 4 |
| 3 | Número de fases elegido a mano | No hay forma de defender por qué 4 y no 3 | Paso 3 |
| 4 | Decenas de datos que en buena parte dicen lo mismo | Ruido que confunde al agrupamiento | Paso 2 |
| 5 | El modelo se califica contra sí mismo, con una sola división de prueba | No sabemos si las fases son reales | Paso 7 |
| 6 | Solo ve precios: no sabe *por qué* se mueven las tasas | El comité necesita la causa, no solo el síntoma | Paso 6 |
| 7 | Muchos datos para pocos episodios históricos | Riesgo de ver patrones que no existen | Pasos 2 y 7 |
| 8 | Una fase puede mezclar episodios distintos (un rally tranquilo y uno de crisis, como 2008 o 2020) | Una etiqueta que agrupa cosas distintas confunde | Paso 7 |

---

## 5. El diseño en 8 pasos

### Paso 1 — Los datos

**Fuente principal: FRED** (Fed de St. Louis), gratis, diario.

| Dato | Para qué |
|---|---|
| Tasas del Tesoro a 3 meses, 1, 2, 3, 5, 7, 10, 20 y 30 años | Construir la curva |

**Para el panel macro (paso 6), fuera del modelo:**

| Dato | Fuente |
|---|---|
| Prima por plazo (modelo ACM) | Fed de Nueva York, gratis |
| Expectativas de la Fed | Futuros de fed funds y SOFR, CME |
| Factor de Cochrane-Piazzesi | Se calcula con las mismas tasas |

> [!warning] Por verificar antes de construir
> Disponibilidad citada de memoria, no verificada. La tasa a 2 años en FRED empezaría en 1976 y la de 30 años tendría un hueco entre 2002 y 2006. Propuesta: arrancar la historia en ~1977, unos 50 años.

### Paso 2 — La foto semanal: pocos datos, bien elegidos

**10 a 15 datos:**

| Grupo | Datos | Qué captura |
|---|---|---|
| Forma de la curva | **Nivel, pendiente, curvatura** (sus 3 componentes principales) | Dónde está la curva y qué forma tiene |
| Velocidad | Cambio en nivel, pendiente y curvatura a 1, 3 y 6 meses | Qué tan rápido se mueve |
| Nervio | Volatilidad de las tasas a 2, 5 y 10 años | Qué tan agitado está el mercado |
| Aceleración | Volatilidad corta menos volatilidad larga | Si el nervio está subiendo o bajando |

**Por qué estos datos:**
- **Nivel, pendiente y curvatura:** en el ejemplo de Dixon explican **95.6%, 4.07% y 0.34%** de la varianza de la curva (*ML in Finance (Dixon et al.) — Part II Sequential*). Con 3 números se resume casi todo.
- **Restar volatilidades:** elimina la redundancia entre datos y agrega una señal de "cambio de ritmo". Es el truco que mejor funcionó en acciones (*Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)*).

**Suavizado.** Cada cambio se promedia con los de ventanas vecinas: el cambio a 8 semanas es el promedio de 7, 8 y 9 semanas. Así, un día de hace dos meses que sale de la ventana no mueve la foto, pero un movimiento de hoy pega completo.

**Regla.** Los datos se calculan diario; el modelo se lee cada semana.

### Paso 3 — Agrupar con un jump model

**La idea en una frase:** repartir 50 años de fotos en pilas de fotos parecidas, **cobrando una multa cada vez que una foto cae en una pila distinta a la de la semana anterior**.

- **Sin multa** (K-means): las fases parpadean.
- **Con multa** (jump model): solo cambia de fase si la evidencia es fuerte. Las fases duran, como en la realidad.

**La multa es una perilla:**
- Muy baja: parpadea.
- Muy alta: reconoce tarde un cambio real.
- Se elige probando.

**Evidencia.** En acciones, el jump model cambió de fase **14 veces** contra **115** del modelo clásico (HMM), con **~5 veces menos rotación** y mejor desempeño (*Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)*).

**¿Cuántas fases?** No se fija a mano:
1. Probar 2, 3, 4 y 5 fases y varias multas.
2. Elegir con **FTIC**, el criterio estadístico que mejor funcionó en estudios con datos simulados (*Generalized Information Criteria for Jump Models (Cortese 2026)*).
3. Confirmar con la prueba de estabilidad del paso 7.

**Si salen 3 fases en vez de 4, es un hallazgo, no un error.** Un modelo más simple es más fácil de defender en comité.

**Retratos.** Al terminar, cada fase tiene un **retrato promedio**: sus valores típicos de nivel, pendiente, volatilidad, etc. Son la base del doble chequeo del paso 5.

**Herramienta.** La librería abierta `jumpmodels` (*jumpmodels (Python Library)*) permite leer la fase de hoy con los parámetros congelados.

### Paso 4 — Confianza real

El modelo en versión continua no dice "fase A", dice **"70% A, 25% B, 5% C"**. La probabilidad sale **directo del agrupamiento**, no de un segundo modelo que lo imita.

| Confianza | Lectura para el comité |
|---|---|
| Alta (>70%) | Fase clara |
| Media (50–70%) | Fase probable, vigilar |
| Baja (<50%) o dos fases empatadas | **Posible cambio de fase en curso** |

*Los umbrales son una propuesta inicial; se calibran en el paso 7.*

### Paso 5 — Trazabilidad: qué datos empujan la decisión

Dos capas que se chequean entre sí.

**Capa A: el ticket (SHAP)**

1. Se entrena un **traductor** (XGBoost) que aprende a dar las mismas respuestas que el jump model.
2. **SHAP** desglosa cada decisión del traductor como un ticket:

> Punto de partida (lo normal): 25%
> \+ volatilidad a 5 años alta: +25
> \+ tasa a 3 años subiendo: +12
> \+ curva 10-30 aplanándose: +8
> **= 70% venta extrema**
>
> *Ilustrativo. Técnicamente SHAP suma en otra escala, pero la lectura es esta.*

**Capa B: comparación contra los retratos (sin XGBoost)**

Se compara la foto de hoy contra el retrato de cada fase. Por ejemplo: "la volatilidad ya está en nivel de venta extrema; la pendiente todavía parece de venta moderada".

**Regla de control:** si la capa A y la capa B cuentan historias distintas, **la lectura se marca como "a revisar"** y no se presenta como concluyente.

**Historial de impulsores.** Un mapa semanal de qué datos ganan o pierden peso con el tiempo. Sirve para detectar cuándo un dato cambia de papel: por ejemplo, la tasa a 10 años pasa de frenar a empujar después de un choque.

> [!note] Advertencia para el comité
> SHAP explica **al modelo**, no al mercado. "La volatilidad empuja" no significa "la volatilidad causa la venta". Para la causa está el panel del paso 6.

### Paso 6 — Panel macro: el "por qué"

Al lado del modelo, **sin entrar al agrupamiento**. Así el modelo conserva ~50 años de historia y su rapidez.

| Indicador | Qué ayuda a explicar |
|---|---|
| **Prima por plazo** | Si las tasas largas suben por riesgo y no por expectativas |
| **Expectativas de la Fed** (futuros) | Si el mercado espera subidas o recortes |
| **Factor de Cochrane-Piazzesi** | Una combinación de tasas forward que explica hasta **44%** del exceso de rendimiento de bonos a 1 año |

Fuente: *Futures-Implied Policy Expectations and Bond Risk Premia*.

**Para qué sirve.** Distinguir dos ventas que el modelo ve iguales. La tasa a 10 años puede llegar al mismo nivel en dos episodios por razones distintas: uno por expectativas de la Fed, otro por prima de riesgo.

### Paso 7 — Probar que las fases son reales

Antes de llevar una sola lectura a comité, el modelo debe pasar 4 pruebas:

| Prueba | Pregunta | Cómo |
|---|---|---|
| **1. Estabilidad** | Si lo entreno con otros años, ¿salen las mismas fases? | Entrenar en sub-periodos distintos y medir cuánto coinciden las etiquetas (índice ARI) |
| **2. Separación** | ¿La tasa a 10 años se mueve claramente distinto en cada fase? | Distribución del cambio de la 10 años a 1 y 3 meses por fase, con rangos, contando **todas** las semanas y no solo las fases que "sobrevivieron" |
| **3. Contra lo obvio** | ¿Dice algo más que "la fase de mañana es la de hoy"? | Comparar contra esa regla y contra "el signo del cambio a 3 meses" |
| **4. Honestidad** | ¿Lo que vemos es real o es suerte? | Bitácora de cada prueba antes de ver el resultado + un periodo reciente que nadie haya visto |

**Detalles de la prueba 4:**
- **Validación móvil:** se reentrena cada 6 meses con solo el pasado y se evalúa en lo que sigue. No hay un solo corte de prueba.
- **Periodo reservado:** los últimos ~2 años (2024–2026) no se tocan hasta el final.
- **Bitácora:** cada configuración probada se registra antes de ver su resultado, siguiendo el protocolo de *Deflated Sharpe Ratio and Backtest Overfitting*.

**Aquí también se confirman las hipótesis de la sección 3:** cuánto duran las fases extremas y si una venta rara vez pasa directo a rally.

**La falla 8 se revisa aquí:** se mide directamente la volatilidad promedio dentro de cada fase. Si una fase mezcla episodios tranquilos con crisis, probablemente hace falta otra fase o datos distintos.

**Criterio de salida:** si el modelo no pasa las pruebas 1 y 2, **no se presenta**. Un termómetro que no es estable no se lleva a comité.

### Paso 8 — Operación

| Reloj | Frecuencia | Qué pasa |
|---|---|---|
| Datos | Diario | Se descargan de FRED y se calculan las fotos |
| **Lectura** | **Semanal** | Fase + confianza, con los parámetros congelados |
| **Alerta** | **Al cambiar de fase** con confianza alta | Aviso al comité sin esperar al reporte mensual |
| **Reporte** | **Mensual** | Ficha al comité (sección 6) |
| **Reentrenamiento** | **Semestral** | Se re-estima todo con la historia nueva. Si las fases cambian mucho, se revisa antes de seguir |

**Por qué no basta con mensual:** si las fases extremas duran semanas (hipótesis de la sección 3), una puede empezar y terminar entre dos comités. Con lectura semanal se ve varias veces.

**Por qué reentrenar solo cada 6 meses:** reentrenar más seguido casi no mejora (en Kaabar, el acierto pasó de 48.55% a 48.92%) y hace que las fases cambien de significado (*Deep Learning for Finance (Kaabar 2024) — Advanced Techniques*).

---

## 6. La ficha mensual al comité (plantilla)

*Ejemplo ilustrativo, sin datos reales.*

> **TERMO — lectura de [mes]**
> **Fase actual:** Venta extrema — **confianza 70%** (segunda opción: venta moderada, 25%)
> **Lleva:** 3 semanas
> **Qué suele venir después** (historia): ~60% se modera a venta moderada en el siguiente mes; rara vez pasa directo a rally
> **Movimiento típico en esta fase:** la tasa a 10 años sube entre X y Y pb en el primer mes (rango histórico)
>
> **Qué la empuja (SHAP):**
>
> | Dato | Efecto |
> |---|---|
> | Volatilidad a 5 años | +++ |
> | Tasa a 3 años subiendo | ++ |
> | Aplanamiento 10-30 | + |
> | Pendiente 2-10 todavía positiva | − |
>
> **Doble chequeo:** coincide con el retrato de la fase ✔
>
> **Contexto macro:**
> - Prima por plazo: subiendo
> - Mercado espera de la Fed: sin cambios
> - Cochrane-Piazzesi: neutral
>
> **Cambios desde el último comité:** pasó de venta moderada a venta extrema el [fecha] (alerta enviada)
>
> **Lo que el modelo NO dice:** hacia dónde irá la tasa, ni qué posición tomar.

---

## 7. Límites y riesgos (para decirlos antes de que los pregunten)

1. **Describe, no predice.** Su utilidad depende de que las fases duren, y eso se mide en el paso 7.
2. **La multa es un balance.** Fases más estables implican reconocer un cambio más tarde. En acciones, el jump model **no detectó el crash de 1987** porque fue demasiado rápido (*Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)*).
3. **El pasado tiene pocos episodios.** 50 años de bonos son pocas fases completas. Cualquier estadística por fase tiene rangos amplios.
4. **Lo que nunca pasó no se reconoce como nuevo.** Un régimen inédito se asigna a la fase más parecida.
5. **SHAP no es causalidad.** Explica al modelo, no al mercado.
6. **La evidencia del jump model es de acciones.** Que funcione igual en bonos es una hipótesis que el paso 7 debe confirmar.

---

## 8. Decisiones abiertas para el comité

1. **¿Qué hace el comité con una alerta?** ¿Solo se informa, o activa una revisión de duración?
2. **Umbrales de confianza** (70% / 50%): ¿se aceptan como inicio y se calibran después?
3. **Número de fases:** si el FTIC elige 3 y no 4, ¿se acepta el modelo más simple?
4. **Historia desde ~1977** (unos 50 años): ¿se acepta esa fecha de inicio a cambio de usar la tasa a 2 años?
5. **Módulo de dirección:** mostrar una predicción de la tasa a 10 años solo en las fases donde haya funcionado. Sería una segunda etapa.
6. **Extensiones del ecosistema:** ¿qué termómetro sigue? ¿Bolsa, materias primas o emergentes?

---

## 9. Fuentes

- Jump models: *Regime-Aware Asset Allocation (Shu Yu Mulvey 2024)*, *Generalized Information Criteria for Jump Models (Cortese 2026)*, *jumpmodels (Python Library)*
- Evaluación honesta: *Deflated Sharpe Ratio and Backtest Overfitting*, *ML in Finance (Dixon et al.) — Part II Sequential*
- Reentrenamiento: *Deep Learning for Finance (Kaabar 2024) — Advanced Techniques*
- Panel macro: *Futures-Implied Policy Expectations and Bond Risk Premia*
- Pendiente: validación cruzada purgada, que solo existe como índice en *Advances in Financial ML (López de Prado) — Book Map*; y la prima por plazo ACM, que no tiene artículo.
