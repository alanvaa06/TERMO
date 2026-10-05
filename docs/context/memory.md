# Memory

> Architecture decisions. One line each. Format: `# decision: sentence`. Cap enforced by hook.

- # decision: primer spec = nucleo go/no-go (M1 datos, M2 features, M3 JM discreto, M8 pruebas 1-3 + lineas base + bitacora); CJM, SHAP, macro, reporte quedan para specs posteriores.
- # decision: L en nivel NO entra al agrupamiento (evita fases por "era de tasas"); S, C, velocidades (incl. de L), vol y 10s30s si; L queda solo como descripcion en reporte.
- # decision: historia desde 1977-02-15 (inicio DGS30); PCA sobre 7 plazos: 1, 2, 3, 5, 7, 10, 30Y; sin variable 10s30s aparte (12 features); DGS3MO fuera (empieza 1981-09-01) y DGS20 fuera (hueco 1987-01 a 1993-09), ambos verificados en CSV de FRED.
- # decision: dato DGS30 tiene dos escalones de ~16-17 pb el 2002-02-19 y el 2006-02-09 (10Y y 20Y casi planas esos dias); entre ambas fechas la serie es de otra construccion, no comparable 1:1 con el resto.
- # decision: DGS30 se usa crudo, sin excluir los 2 dias de escalon del PCA (aportan ~0.1% de varianza, estimado); 2002-02-19 a 2006-02-08 queda como limitacion documentada.
- # decision: ventana de entrenamiento expansiva desde 1977; volatilidad y aceleracion en logaritmos (log sigma, log de cocientes) para que la era Volcker no domine el z-score.
- # decision: K y lambda se eligen por puntaje walk-forward (estabilidad x separacion) sobre grilla de 24 configs registrada antes de correr; FTIC (Fan-Tang, Cortese et al.) solo como segunda opinion sobre K; si discrepan gana el K menor.
- # decision: FTIC no sirve para elegir lambda en JM no disperso: minimizarlo equivale a lambda ~ a_T*K0/2 (~8 con T~12400, p=12), fijado por formula (derivacion propia, sin probar).
- # decision: preproceso = recorte a +/-3 desviaciones y luego z-score, ambos con estadisticas de la ventana de entrenamiento (clases DataClipperStd y StandardScalerPD de jumpmodels).
- # decision: el JM se entrena con datos diarios; la lectura semanal es el ultimo estado de la inferencia en linea sobre la secuencia diaria.
- # decision: separacion = eta2 en exceso sobre un nivel de azar por VOLTEO DE SIGNOS en bloques de 26 semanas (no por desplazamiento de fases: ese inflaba +0.005 a +0.010 cuando una fase rara coincide con alta volatilidad). Con eta2 crudo, 5 fases de ruido ganaban a 2 fases de ruido 25% de las veces (nominal 2.5%).
- # decision: la medicion por desplazamiento se registra y reporta como comparacion declarada de antemano; nunca decide. No se cambia de criterio despues de ver un resultado: las alternativas se calculan en la misma corrida.
- # decision: cada etapa (run, report, final-holdout) esta amarrada a la configuracion completa, el hash del snapshot y el commit del registro; reportes y resultado de holdout van a la bitacora; registrar exige codigo commiteado.
- # decision: el ganador exige estabilidad y separacion positivas; sin ninguno, no-go. La misma regla aplica al modelo mas simple que propone FTIC.
- # decision: la identidad del codigo es el hash de contenido de git para src, configs y pyproject.toml (no el commit ni la carpeta actual); tambien se amarran las versiones de librerias y el SHA-256 de cada archivo de etiquetas.
- # decision: el usuario mantuvo los criterios tal cual (intervalo de 95%) sabiendo la potencia; registro hecho el 2026-10-02 con 30 trials y 10 variables (la regla de colinealidad quito logr2_20_60 y logr10_20_60 en la primera ventana).
- # decision: potencia medida del criterio contra la inercia: detecta eta2=0.01 el 15%, 0.02 el 40%, 0.05 el 94% de las veces. Un no-go significa "no se distingue con claridad de la inercia", no "no hay fases".
- # decision: prueba de independencia = chi2 con valor p por TODOS los desplazamientos circulares; excluir los pequenos rechaza 4% bajo la nula al nivel 1%.
- # decision: linea base K-means con sklearn.KMeans (mismo objetivo que JM con multa 0; jumpmodels tarda ~84 s por ajuste con multa 0).
- # decision: jumpmodels 0.1.1 verificado en Python 3.14 / numpy 2.5 / pandas 3.0 / sklearn 1.9; predict_online causal; ajuste K=5, T=12000 ~1 s.
- # decision: flujo en 5 etapas CLI (snapshot, register, run, report, final-holdout); bitacora trials/trials.jsonl append-only; final-holdout corre una sola vez y requiere aprobacion del usuario.
- # decision: experimento 2 = receta de datos TYCCLES (139 rangos causales de 126/252 dias sobre cambios suavizados a 21-189 dias de 7 plazos y 4 medidas de curva, mas vol 21d), sin regla de colinealidad, mismas pruebas y umbrales; registro aparte en trials/exp2 y reports/exp2.
- # decision: la multa del JM se registra POR VARIABLE (lambda = c * p, c en {0.5..50}); con p=10 es la grilla de spec 1. Sin esto, 139 variables dejarian el mismo lambda 14 veces mas debil.
- # decision: K-means congelado (ajuste unico hasta 1997-12-31, lee 1998 -> 2024-09) es familia aparte: estabilidad = S2 sola; cada familia tiene su ganador y su tabla; GO si una familia pasa; la elegida es la de mayor limite inferior de separacion.
- # decision: las variables se calculan con una 'receta' intercambiable (PcaRecipe = spec 1, TycclesRecipe); los rangos se redondean a 6 decimales en pb antes de rankear para que movimientos iguales empaten.
- # decision: potencia del criterio de separacion a 1,390 semanas (familia congelada): eta2 0.01 -> 8%, 0.02 -> 28%, 0.05 -> 83%. Un no-go de esa familia es evidencia debil.
- # decision: --trials-dir y --reports-dir van juntos (uno solo se rechaza); cambiar codigo deja cerradas las etapas pendientes del registro de spec 1.
- # decision: dos experimentos registrados (z-scores PCA y rangos TYCCLES) dan el mismo resultado: separacion maxima ~0.012-0.014 del movimiento a 4 semanas, no distinguible de la inercia; la receta de datos no era la causa. El usuario mantiene el holdout estricto (abierto sin resultado = perdido).
- # decision: (2026-10-05, usuario) TERMO sigue como herramienta SOLO DESCRIPTIVA: jump model sobre la receta de rangos + SHAP para explicar; se declara por escrito que no anticipa el 10Y mejor que la inercia (62 trials). HSBC hace el mismo descargo ('contemporaneous model, does not predict') pero sus graficas condicionan a que la fase sobreviva; TERMO no publica ese tipo de grafica. Es decision de producto nueva, no cambio retroactivo de criterio.
- # decision: el holdout (2024-10 -> 2026-09) quedo GASTADO el 2026-10-05 por la prueba descriptiva; cualquier validacion futura es modo sombra (lecturas en vivo). Resultado: fases persistentes y coherentes en direccion; la etiqueta 'rally de la parte larga = aplana' no se sostuvo en ese periodo.
