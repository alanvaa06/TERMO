# Results

> Build log. 1-4 lines per finished item. List format. Older detail lives in `archive/`. Cap enforced by hook.

- 2026-10-01 spec 1 (nucleo go/no-go) escrito en docs/superpowers/specs/2026-10-01-nucleo-go-no-go-design.md; pendiente revision del usuario y plan de implementacion. Sin codigo todavia.
- 2026-10-02 segunda revision independiente de los arreglos: sin look-ahead, criterio de separacion conservador (pasa por suerte 0.1%-2.0%, nominal 2.5%); 5 huecos de integridad/pruebas cerrados en 4c5f149. 184 pruebas, 27 errores sembrados detectados. Snapshot FRED 2026-10-02 validado (sin huecos desde 1977-02-15).
- 2026-10-01 tareas 1-23 del plan ejecutadas en feat/nucleo-go-no-go (22 con subagentes, cada archivo identico al prototipo). Revision independiente: sin look-ahead; 9 hallazgos, todos arreglados (commits d2ba5b1, fd4dc40). 167 pruebas, ruff y mypy limpios, 11 errores sembrados detectados.
- 2026-10-01 plan de implementacion escrito; su codigo se prototipo fuera del repo con datos sinteticos: 140 pruebas, ruff y mypy limpios, orden TDD reproducido tarea por tarea. Nada corrido sobre datos reales.
- 2026-10-01 al prototipar se corrigieron 3 puntos del spec: separacion en exceso, independencia con todos los desplazamientos, K-means con sklearn. En la curva sintetica el veredicto fue no-go (inercia dificil de batir con lectura en linea).
- 2026-10-01 verificado en FRED (filas CSV): DGS3MO desde 1981-09, DGS2 desde 1976-06, DGS30 desde 1977-02-15 con escalones 2002-02-19/2006-02-09, DGS20 sin datos 1987-01 a 1993-09.
- 2026-10-01 leido codigo de jumpmodels 0.1.1: predict_online causal, falla con NaN, sin seleccion de hiperparametros, CJM impracticable con K=5 a rejilla 5%.
