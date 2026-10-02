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
