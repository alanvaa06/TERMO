# Lessons

> Rules from corrections. Friction only — never repeat a logged mistake. List format. Cap enforced by hook.

- Todo test estadistico que sea criterio de go/no-go se calibra por simulacion bajo la nula ANTES de ir al spec. Caso: dos criterios del spec aprobado (eta2 crudo contra inercia, desplazamiento circular con zona de exclusion) rechazaban de mas; se detecto solo al prototipar.
- Disponibilidad de series: verificar en las filas del CSV, nunca en la nota/metadata de la pagina. Caso: nota de FRED dice DGS30 "discontinued 2002-2006" pero el CSV diario tiene valores en todo ese periodo; reporte un hueco inexistente.
- Contar trials: el registro 'setup' no es un trial (spec 1 = 30 registros = 29 trials). Antes de escribir un total en un spec, contarlo con el mismo codigo que lo imprimira.
- En un plan, `A + """...""".replace(...)` aplica el replace solo al literal final; poner parentesis. Los subagentes lo detectaron al correr la prueba.
