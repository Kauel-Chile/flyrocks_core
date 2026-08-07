# Roadmap Flyrocks — próximas 6 semanas

> **Borrador para completar con el equipo.** Fecha: 2026-08-05.
> Todo lo marcado ⟨así⟩ o `[ ]` es un espacio a rellenar.
> Versión interna. La versión para el cliente sale de aquí, recortada (ver §7).

---

## 1. El mensaje (esto es lo que hay que dejar claro antes de mostrar fechas)

Tres frases, en este orden. Si el cliente se queda solo con esto, basta:

1. **El resultado está asegurado desde ya.** La herramienta de edición sobre el
   canvas permite obtener la máscara de trayectorias completa **hoy**, sin
   depender de que el algoritmo resuelva los casos difíciles. No hay escenario en
   que el cliente se quede sin entregable.
2. **Lo que estamos optimizando es el esfuerzo manual, no el resultado.** Cada
   avance algorítmico se traduce en *menos trazos que el operador tiene que
   dibujar a mano*. Ese es el número que vamos a reportar cada quincena.
3. **La parte automática son hipótesis con fecha de corte.** Cada línea de
   investigación tiene criterio de éxito y plazo. Si no rinde, se cierra y se
   pasa a la siguiente — no se arrastra. Hay ⟨N⟩ líneas en cola, así que el
   avance no depende de que una sola funcione.

**Regla de redacción para el cliente:** nunca escribir "vamos a resolver X".
Escribir "vamos a **probar** X; si a la fecha Y no supera ⟨umbral⟩, cerramos y
seguimos con Z". Prometer el *proceso*, no el *resultado*. Suena más profesional
y además es lo que realmente podemos cumplir.

---

## 2. La métrica que hace visible el avance

Hoy el problema de percepción es que **cada semana se ve igual**: imágenes de
trayectorias que mejoran de un modo que solo nosotros apreciamos. Con un número,
el avance se vuelve innegable.

**Métrica propuesta: "esfuerzo de edición"** — sobre una tronadura de referencia:

| Indicador | Cómo se mide | Basal ⟨llenar⟩ | Meta S6 ⟨llenar⟩ |
|---|---|---|---|
| Trayectorias correctas sin tocar | conteo | ⟨ ⟩ | ⟨ ⟩ |
| Trayectorias que hubo que **unir** a mano | clics de unión | ⟨ ⟩ | ⟨ ⟩ |
| Trayectorias que hubo que **borrar** (falsos por humo) | clics de borrado | ⟨ ⟩ | ⟨ ⟩ |
| Trayectorias que hubo que **dibujar de cero** | trazos nuevos | ⟨ ⟩ | ⟨ ⟩ |
| **Tiempo total de edición** | minutos | ⟨ ⟩ | ⟨ ⟩ |

> **Prerrequisito (S1, no negociable):** el editor debe registrar estas acciones.
> Es ~medio día de trabajo y convierte cada sesión de uso en una medición.
> Sin esto, "mejoramos" es una opinión.

`[ ]` Decidir con el equipo: ¿una tronadura de referencia o dos? ⟨ ⟩
`[ ]` ¿El cliente acepta que la métrica sea "minutos de edición"? ⟨ ⟩

---

## 3. Los tres carriles

Se trabajan **en paralelo**, y eso es lo que hay que mostrar: no estamos
apostando todo a que el algoritmo funcione.

| Carril | Qué es | Naturaleza | Riesgo |
|---|---|---|---|
| **A — Herramienta** | Editor de trayectorias, usabilidad, reporte | **Compromiso.** Alcance conocido, se entrega. | Bajo |
| **B — Algoritmo** | Reducir el trabajo manual | **Exploratorio.** Hipótesis con corte. | Alto por ítem, bajo en conjunto |
| **C — Base** | Velocidad de iteración, medición | Habilitador | Bajo |

---

## 4. Gantt — 6 semanas (S1 arranca lunes 10-ago)

```
                        S1        S2        S3        S4        S5        S6
                     10-14ago  17-21ago  24-28ago  31a-4sep  07-11sep  14-18sep
CARRIL A — HERRAMIENTA (compromiso)
A1 Editor: unir/cortar  ████████
A2 Editor: crear/borrar ████████
A3 Deshacer + sesión              ████████
A4 Telemetría de uso    ████
A5 Pintado de zonas web           ████████  ████
A6 Malla/homografía web                     ████████  ████
A7 Usabilidad y pulido                                ████████  ████████
A8 Reporte PDF final                                            ████████  ████
                             ▲DEMO 1           ▲DEMO 2            ▲ENTREGA

CARRIL B — ALGORITMO (hipótesis, con corte)
B1 Recalibrar capas     ████
B2 Puente máscara→track ████████  ████
B3 Asociación global              ████████  ████████  ████
B4 Consenso / ensemble                                ████████  ████
B5 Rescate de curvas                                            ████████
                              ▲corte B2         ▲corte B3          ▲corte B4

CARRIL C — BASE
C1 Caché de pipeline    ████████
C2 Set de referencia    ████
C3 Medición continua      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

`[ ]` Ajustar semanas según capacidad real del equipo: ⟨quién toma cada carril⟩
`[ ]` ¿Hay fecha comprometida con el cliente que obligue a comprimir esto? ⟨ ⟩

---

## 5. Detalle por ítem

### Carril A — Herramienta (esto se entrega sí o sí)

| ID | Qué | Semana | Listo cuando… |
|---|---|---|---|
| **A1** | Unir dos segmentos y cortar uno en un punto | S1 | el operador reconstruye una parábola partida en 2 con 1 clic |
| **A2** | Crear trayectoria a mano libre y borrar | S1 | se dibuja una roca que el algoritmo no vio, sobre el canvas gris |
| **A3** | Deshacer/rehacer + guardar y retomar sesión | S2 | se cierra el navegador y se recupera el trabajo |
| **A4** | Telemetría de edición (§2) | S1 | cada sesión deja un JSON con los conteos |
| **A5** | Pintar zonas (verde/rojo/azul) en la web | S2–S3 | ya no se usa Paint ni PNG externos |
| **A6** | Colocar/rotar la malla de tiros en la web | S3–S4 | la homografía sale de la UI, no de un script |
| **A7** | Usabilidad: atajos, zoom, rendimiento con miles de trazas | S4–S5 | ⟨definir con el operador tras Demo 1⟩ |
| **A8** | Reporte PDF con distancias y zonas | S5–S6 | ⟨definir contenido con el cliente⟩ |

`[ ]` Falta priorizar A7 con feedback real. **Reservar la lista para después de
la Demo 1** — es más creíble decirle al cliente "esto lo definimos con tu
operador" que llegar con una lista inventada. ⟨ ⟩

### Carril B — Algoritmo (hipótesis; se cierran, no se arrastran)

Formato obligatorio de cada línea: **hipótesis → criterio de éxito → corte → qué
pasa si falla.**

| ID | Hipótesis | Criterio de éxito | Corte | Si falla |
|---|---|---|---|---|
| **B1** | Recalibrar los umbrales de las capas (P2/P4) deja la máscara utilizable | ⟨~49 de 68 trazos, >90% en zona de interés⟩ | S1 | se congela la calibración actual y se sigue con B2 |
| **B2** | La máscara de forma (línea vs mancha) mejora al tracker si se acumula en ventanas cortas (P3) | ⟨reducción de X% en falsos por humo⟩ | S2 | se usa solo como ROI espacial (opción simple) y se pasa a B3 |
| **B3** | La **asociación global** de tracklets (P7) une lo que hoy queda fragmentado | ⟨reducción de X% en uniones manuales⟩ | S4 | queda como unión asistida en el editor (el operador une, el sistema sugiere) |
| **B4** | El **consenso** entre máscaras (P6) filtra falsos sin perder rocas reales | ⟨menos borrados manuales, sin perder ninguna azul⟩ | S5 | se descarta; el filtrado queda manual |
| **B5** | El **rescate por trazo azul** (P5) reconstruye las curvas que hoy se pierden | ⟨las 6 curvas de referencia reconstruidas⟩ | S6 | el operador las dibuja a mano en el editor (A2) — ya cubierto |

> **Lo importante de esta tabla:** la columna "Si falla" nunca dice "no hay
> solución". Siempre cae en la herramienta manual, que ya está construida. **Eso
> es lo que transmite tranquilidad**, no las promesas del resto de la tabla.

`[ ]` Llenar los ⟨X%⟩ con el equipo. Si no hay basal medido aún, poner
"se define al cerrar C2 (S1)" — es preferible a inventar un número.

### Carril C — Base

| ID | Qué | Semana | Por qué importa |
|---|---|---|---|
| **C1** | Caché por nodo del pipeline (P1) | S1 | hoy probar una idea cuesta reprocesar todo el video; con esto las iteraciones bajan de ⟨horas⟩ a ⟨minutos⟩ → **más hipótesis probadas por semana** |
| **C2** | Set de referencia + basal de la métrica (§2) | S1 | sin esto no se puede afirmar que algo mejoró |
| **C3** | Medición en cada demo | continuo | la curva de "minutos de edición" bajando es el mejor gráfico para el cliente |

---

## 6. Hitos y cadencia

| Fecha | Hito | Qué se muestra |
|---|---|---|
| Vie 21-ago | **Demo 1** | Editor funcionando de punta a punta: el operador toma una tronadura real y produce la máscara final. Se muestra el **basal** de esfuerzo. |
| Vie 04-sep | **Demo 2** | Zonas y malla en la web + resultado de B2/B3. Se muestra la **variación** de esfuerzo vs Demo 1. |
| Vie 18-sep | **Entrega** | Flujo completo + PDF. Curva de esfuerzo de las 3 mediciones. |

Entre demos: **nota semanal de 5 líneas** (qué se probó, qué resultó, qué sigue).
Es barato y elimina la sensación de silencio, que es la mitad de la inquietud del
cliente.

`[ ]` ¿Demos por videollamada con el operador presente? Recomendado: que **él**
maneje la herramienta en la demo, no nosotros. ⟨ ⟩

---

## 7. Qué recortar para la versión del cliente

- **Sacar:** nombres de archivos, IDs internos (P1–P10), detalle de por qué
  fallaron los intentos anteriores.
- **Mantener:** §1 (mensaje), §2 (métrica), el Gantt de §4 simplificado a los
  tres carriles, §5-B **solo con las columnas Hipótesis / Cuándo sabremos**, y §6.
- **Agregar:** una lámina con la captura del editor. Ver la herramienta calma más
  que cualquier tabla.
- **Tono:** "seguimos trabajando en el proyecto" no se dice, se muestra con la
  cadencia de §6.

---

## 8. Riesgos a declarar de frente (no esconderlos)

Declarar un riesgo antes de que ocurra es señal de control; explicarlo después es
señal de improvisación.

| Riesgo | Impacto | Mitigación (esta es la parte que tranquiliza) |
|---|---|---|
| Ninguna línea de B rinde lo esperado | El operador edita más de lo deseable | La herramienta ya entrega el resultado completo; el esfuerzo es acotado y medido |
| Hay trayectorias irrecuperables por software | Se pierden casos límite | Rescate manual sobre el canvas (A2) — cubierto desde S1 |
| El operador no se acomoda a la herramienta | Adopción lenta | Demo 1 la maneja él; A7 se prioriza con su feedback |
| ⟨ ⟩ | ⟨ ⟩ | ⟨ ⟩ |

---

## 9. Preguntas para resolver mañana con el equipo

1. ¿Hay una fecha contractual? Cambia si el plan es de 6 semanas o de ⟨ ⟩.
2. ¿Quién toma cada carril? A y B en paralelo requieren ⟨N⟩ personas.
3. ¿Cuál es la tronadura de referencia y quién la edita para el basal?
4. ¿La audiencia de la presentación es gerencia o técnica? (Gerencia → §1 + §6
   + curva de esfuerzo. Técnica → agregar §5-B completa.)
5. ¿Mostramos la tabla de riesgos (§8)? Recomendación: **sí**, en versión corta.
6. ¿Qué otras tronaduras hay disponibles para validar que esto no está
   sobreajustado a un solo pozo?
