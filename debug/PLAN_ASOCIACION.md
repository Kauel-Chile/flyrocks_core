# Plan por etapas — Asociación trayectoria → tiro de origen

> **Qué resuelve:** hoy entregamos trazos que no dicen *de dónde salió la roca*.
> El cliente necesita que cada trayectoria apunte a su **pozo de origen**, que es
> lo que hace su versión actual cruzando malla de detonación + tiempos.
>
> **Estrategia:** primero la proyección **2D** sobre trazos dibujados a mano
> (E1–E4 = la demo). El calce **temporal** viene después, sobre detecciones del
> pipeline (E6–E7). Creado: 2026-08-06.
>
> Relacionados: `debug/PARABOLAS.md` (por qué el origen visible no está sobre el
> pozo) · `debug/PENDIENTES.md` §P5, §P7 · `debug/homografia.py` · `pre_tiros.py`

---

## 0. Los números que gobiernan el diseño

Medidos sobre `Secuencia (2).csv` y la homografía vigente. **No son supuestos.**

| Dato | Valor |
|---|---|
| Pozos con tiempo válido | **113** (tiempos de detonación **todos únicos**) |
| Extensión de la malla | 64 × 92 m → 549 × 789 px |
| Separación entre pozos vecinos | **5.45 m = 47 px** (mediana) |
| Escala | 8.58 px/m · encuadre 3840 px = 448 m |
| Ventana de detonación | 3000–5769 ms = 2.77 s = **83 frames** |
| **Δt entre pozos espacialmente vecinos** | **216 ms = 6.5 frames** (mediana) |
| Pares dentro de 1 frame | 148 de 6328, separados 19.7 m |
| Video | 29.97 fps · 453 frames · inicio de tronadura = **frame 48** |

**La conclusión de la que cuelga todo el plan:** geometría y tiempo son
**complementarios casi perfectos**. Los pozos que el espacio no distingue (47 px)
detonan separados por 6.5 frames. Los que el tiempo no distingue (<1 frame) están
a 19.7 m. Cada fuente cubre el punto ciego de la otra — por eso E1–E4 dan
*candidatos* y solo E7 puede dar *el pozo*.

### Las dos fuentes de error, cuantificadas

**(a) Error angular del trazo — es el dominante.** Al prolongar hacia atrás:

| Largo de la traza | error 1° | error 3° | error 5° |
|---|---|---|---|
| 100 m | 0.3 pozos | **1.0 pozos** | 1.6 pozos |
| 150 m | 0.5 pozos | 1.4 pozos | 2.4 pozos |
| 200 m | 0.6 pozos | 1.9 pozos | 3.2 pozos |

→ **La región de candidatos es una cuña que se abre con la distancia**, no un
círculo de radio fijo. Una traza corta puede resolver el pozo; una de 200 m
honestamente no puede, y la interfaz debe mostrarlo.

**(b) Paralaje del origen — sesgo sistemático, no ruido.** El inicio visible está
desplazado **radialmente hacia afuera** del nadir (ver `PARABOLAS.md` §2). A 100 m
del nadir la corrección vale 2–8 m según la altura de la roca al hacerse visible:
**medio pozo a un pozo y medio**, y siempre en la misma dirección.

### El parámetro `k` (y por qué NO necesitamos la altura del dron)

En la corrección, la altura del dron `h` y la altura de la roca `z` entran siempre
juntas y colapsan en **un solo escalar adimensional**:

```
   origen_corregido = nadir + (p0_observado − nadir) / (1 + k)
   con   k = z / (h − z)        rango útil ≈ 0.02 – 0.10
```

Se calibra **con un slider**, mirando plausibilidad. Descartado depender del
`.SRT` del dron: su altura es relativa al **punto de despegue**, y en un rajo con
el operador en otra cota ese número miente por decenas de metros.

> Método objetivo para calibrar `k` **sin ground truth** (ver E4): con `k` muy
> chico los pozos elegidos se amontonan en el **borde exterior** de la malla; con
> `k` muy grande, en el **centro**. El `k` correcto es el que hace que la
> distribución de pozos elegidos **se parezca a la distribución de la malla**.

---

## FASE A — La demo (E1 a E4). Solo trazos dibujados a mano.

Decisión tomada: sin cargar detecciones del pipeline. Con trazos manuales el
error de detección es **cero**, así que si el match falla el culpable es la
matemática de asociación y nada más. Es el experimento limpio.

---

### E1 — La malla sobre la imagen ✅ **hecha 2026-08-06**

**Objetivo:** ver los 113 pozos proyectados encima de la máscara, bien calzados.
Sin asociación todavía.

**Implementado en:** `debug/malla_export.py` (exporta e inyecta) +
`debug/demo/trayectorias.html` (dibuja, ajusta, exporta).

- **No hubo que recalibrar nada.** Se reusó `05_tiros/h_matrix.json` tal cual
  (RMS 3.19 px, anclada al frame_69). Verificado sobre `1_intensidad.png`: los
  113 pozos caen sobre la zona del blast y el gradiente de color reproduce la
  propagación **derecha→izquierda** ya documentada. Control independiente en
  `out/6_mascaras/verif_malla.png`.
- **Datos embebidos, no `fetch`:** el visor se abre con doble clic y `file://`
  bloquea las peticiones locales. `malla_export.py` inyecta el bloque entre
  marcadores; para recalibrar, se corre de nuevo.
- **Color:** rampa **ordinal de un solo tono** (azul, pasos 100→600), validada
  monótona con el extremo oscuro a 2.59:1 contra el negro de la máscara.
  Descartado el JET de `homografia.py`: un arcoíris inventa bandas donde no las
  hay. Cada pozo lleva **anillo oscuro** — sin esa referencia local de contraste,
  la máscara (que va de negro a blanco) se come los extremos de la rampa.
- **Ajuste fino:** similitud (mover / girar / escalar) sobre la proyección ya
  calibrada, no re-resolución de la homografía — la deriva medida del dron es
  esencialmente traslación (§P9). Durante el ajuste **el dibujo queda
  suspendido**, para no dejar trayectorias basura al encajar la malla.
- **Extra:** el PNG exportado incluye la malla si está encendida (mismo principio
  que ya regía el visor: sale lo que se ve).

**Qué se hace**
1. Cargar `h_matrix.json` (ya existe, RMS 3.19 px vía `pre_tiros.py`) y el CSV de
   secuencia. Proyectar los 113 pozos a píxeles con el afín.
2. Dibujarlos como puntos, con etiqueta al acercar y **color por
   `DetonatingTime`** (mismo colormap que `homografia.py`): de una se ve la
   propagación derecha→izquierda y el cliente reconoce su propia secuencia.
3. **Ajuste fino** de la malla: arrastrar / rotar / escalar el conjunto, o mover
   2–4 anclas con recálculo del afín por mínimos cuadrados en JS (~30 líneas,
   mismo `lstsq` de `affine_from_correspondences`).

**No reimplementar el ajuste desde cero** — `pre_tiros.py` ya lo hace con zoom.
Acá solo va el retoque.

**Entregable:** la malla encima de la máscara, ajustable, con la secuencia
visible por color.
**Aceptación:** los pozos caen sobre los cráteres visibles en la máscara; el
cliente reconoce su malla.
**Riesgo:** bajo. Es la etapa segura y ya vale como avance mostrable.

---

### E2+E3 — RESULTADOS MEDIDOS (2026-08-06)

Implementadas juntas: E2 sola no se puede evaluar (muestra una flecha de
corrección y nada más). Banco de pruebas: `debug/test_asociacion.py`, que replica
la función `asociar` del visor y la corre contra trayectorias sintéticas con
origen conocido — 113 pozos × repeticiones, variando cuánto tapa el humo el
arranque y cuánto se equivoca el trazo a mano.

**Rendimiento con `k` bien calibrado: top-1 = 88 %, top-3 = 99 %.**
El pozo verdadero **nunca salió del top-5** en ninguna corrida.

| | 0° err | 2° err | 5° err | 10° err |
|---|---|---|---|---|
| top-1 | 88 % | 86 % | 78 % | 73 % |
| top-3 | 100 % | 100 % | 99 % | 97 % |

**La apertura de cuña óptima es 4–6°** (a 2° cae a 65 %, a 20° cae a 56 %). El
default de 4° queda confirmado por medición, no por gusto.

#### ⚠️ Corrección a §0: el paralaje NO es de segundo orden

Lo dije al empezar E2 mirando solo el centro de la malla (12.8 m del nadir) y
**estaba equivocado**. El punto que importa no es el centro: es dónde cae el
inicio del trazo, que está más afuera — los pozos del borde llegan a 60 m del
nadir y el arranque visible suma 20 m más. La matriz *paralaje real* × *paralaje
asumido* lo deja sin discusión:

| top-1 | k asumido 0.00 | 0.04 | 0.08 | 0.15 | 0.25 |
|---|---|---|---|---|---|
| **k real 0.00** | **88 %** | 78 % | 46 % | 26 % | 19 % |
| **k real 0.08** | 55 % | 83 % | **88 %** | 56 % | 25 % |
| **k real 0.25** | 19 % | 23 % | 27 % | 52 % | **88 %** |

La diagonal siempre da 88 %; **cada paso que te equivocas en `k` cuesta entre 8 y
40 puntos**. `k` es el parámetro más sensible de todo el sistema, no un ajuste
fino. El slider es obligatorio y hay que calibrarlo con datos reales.

#### El detector de `k` sin ground truth: funciona a medias

La idea de §0 era comparar la distribución radial de los pozos elegidos contra la
de la malla. Medido:

- Con `k` **sobreestimado** el sesgo se vuelve **negativo** — se detecta limpio.
- Con `k` correcto o **subestimado** el sesgo se queda plano en ≈ +1.5 m — **no
  distingue entre ambos casos**.

O sea sirve como **cota superior** de `k`, no como calibrador fino. El cruce por
cero ocurre ≈ 0.05 por encima del `k` verdadero. Es señal útil, pero no
reemplaza calibrar contra trayectorias con origen conocido.

#### La ambigüedad irreducible (y por qué confirma el plan)

Cuando falla, **el 67 % de las veces elige un pozo más externo que el verdadero**,
a 8.0 m de mediana (1.5 pozos). La causa es estructural: los pozos **alineados
sobre el eje de vuelo** son indistinguibles con geometría sola, y el algoritmo
apuesta por el más cercano al inicio del trazo.

Esto es exactamente lo que anticipaba §0, y el tiempo lo resuelve de raíz: esos
pozos alineados detonan separados por **216 ms = 6.5 frames**. No es una
limitación del algoritmo — es información que no está en la imagen.

**Conclusión operativa:** reportar candidatos, no pozo único. La decisión que ya
estaba tomada queda respaldada por medición.

---

### E2 — Nadir y paralaje ✅ **hecha** (fusionada en E3)

Nadir arrastrable (arranca en el centro del cuadro, visible solo en modo
**Ajustar…**) y slider `k` de 0 a 0.15. El origen corregido se dibuja unido al
punto dibujado por un segmento, con etiqueta al acercar, y el pie informa cuántos
**metros** corrió el paralaje.

**`k` arranca en 0 a propósito.** Mientras no esté calibrado con datos reales,
corregir con un valor inventado mueve el origen sin fundamento. Con `k = 0` el
origen es exactamente el punto dibujado — una variable menos al validar.

> Se fusionó con E3 porque **E2 sola no se puede evaluar**: muestra un
> desplazamiento y nada con qué contrastarlo.

---

### E3 — Asociación: la cuña de retroceso ✅ **hecha**

**Algoritmo final** (función `asociar` en el visor, portada a
`debug/test_asociacion.py`), por trayectoria:

1. **Origen corregido:** `O = nadir + (p0 − nadir) / (1 + k)`.
2. **Dirección de retroceso:** opuesta a la tangente inicial de la Bézier
   (`p0 → c1`).
3. **Costo por pozo**, con `a` = avance hacia atrás y `perp` = desvío del eje:

   | Término | Regla |
   |---|---|
   | **Causalidad** | pozo *adelante* del inicio (`a < −8 m`) → descartar |
   | **Alineamiento** | `perp / (tan σ · max(a, 15 m))` — la tolerancia **se abre con la distancia** |
   | **Lejanía** | `a / 70 m` — la roca se hace visible cerca del pozo, no a 80 m |

4. **Ranking:** 5 mejores, confianza relativa por softmax del costo.

**σ = 4° confirmado por medición** (a 2° cae a 65 %, a 20° a 56 %).

#### ❌ Descartado: la "segunda recta" `nadir → p0`

El blast de este video cae a **12.8 m del centro del cuadro**, o sea casi bajo el
dron. Esa recta queda casi colineal con la dirección de vuelo y no aporta nada.
La degeneración que había marcado como excepción **es el caso general aquí**.

---

### E4 — Confianza y degradación ✅ **hecha** (el diagnóstico de `k`, a medias)

**Tres niveles**, por cuánto destaca el mejor candidato: **pozo** (conf ≥ 0.55) ·
**grupo** (≥ 0.28) · **sector**. El pie los nombra distinto (`origen:` /
`origen probable:` / `zona:`) y el JSON exporta el nivel.

**Candidatos** dibujados con anillo naranja, grosor y opacidad según confianza
(naranja contra la rampa azul de los pozos = las dos primeras ranuras
categóricas, el par más separado disponible).

**El diagnóstico automático de `k` quedó a medias** — ver §E2+E3: detecta
sobreestimación, no distingue entre correcto y subestimado. Sirve como cota
superior. **Queda pendiente** calibrar `k` contra trayectorias con origen
conocido.

---

### E4b — Completar la trayectoria hasta el pozo ✅ **hecha** (no estaba en el plan)

**Pedido del cliente, textual:** *"que la trayectoria salga y se dibuje desde el
pozo más plausible"*. No quiere que le señalen el pozo — quiere ver la
trayectoria **naciendo** en él.

- Tramo Bézier cuadrática del pozo al inicio del trazo, con el punto de control
  apoyado en la **misma tangente** con que arranca la curva dibujada: el empalme
  no tiene quiebre y se lee como una sola trayectoria.
- **Mismo color, punteado**, con aro en el pozo. Otro color lo haría leer como
  otra entidad; el punteado dice "este tramo no se ve en el video, lo puso el
  sistema" sin romper la unidad ni aparentar certeza que no hay.
- Se dibuja para **todas** las trayectorias, no solo la seleccionada.
- Entra al PNG exportado; el JSON incluye `punto_px` del pozo elegido.

Verificación visual: `out/6_mascaras/verif_completado.png`.

---

## FASE B — El calce temporal (E5 a E7). Después de la demo.

---

### E5 — Arquitectura de dos fuentes

**Objetivo:** dejar la puerta abierta sin usarla todavía.

Normalizar el formato interno para que una trayectoria pueda venir de **dibujo
manual** o de **detección del pipeline**, con campos opcionales `t_inicio` /
`t_fin` y un campo `fuente`. El asociador consume el formato, no la procedencia.

**Nota:** las detecciones actuales del pipeline propio **difieren** de las del
equipo del cliente. Por eso esta etapa deja el contrato listo pero no depende de
cuál pipeline gane.

---

### E6 — Herencia temporal: el puente entre las dos fases

**El problema:** la máscara es **acumulada**, no tiene eje temporal. Un trazo
dibujado a mano no trae `t`.

**La solución:** las detecciones del pipeline sí traen `[id, x, y, t]`
(`polar_result.npz`). Una curva dibujada puede **heredar el tiempo** de las
detecciones que caen dentro de su corredor. **La maquinaria ya existe**:
`exp_azul_contenidas.py` calcula exactamente "qué trayectorias caen
mayoritariamente dentro de este trazo" (el discriminador correcto es la fracción
del largo, no el simple contacto).

**Por qué importa doble:** es el mismo mecanismo de rescate de las trayectorias
azules (§P5). Una etapa, dos problemas.

**Riesgo conocido:** en zonas donde el dedup agresivo borró todo (caso azul#2, 0
pedazos) no hay de quién heredar. Hay que ir al **set rico**, no a `1_dedup`.

---

### E7 — Match espacio-temporal completo

**Objetivo:** pasar de *candidatos* a *el pozo*.

Con `t` disponible, entran dos restricciones nuevas, y el problema queda
**sobredeterminado** (4 observaciones contra 1 incógnita):

1. **Causalidad dura:** el pozo debe haber detonado **antes** del nacimiento de la
   traza. Sin parámetros que calibrar, elimina candidatos gratis.
2. **Tiempo de vuelo coherente:** `t_nacimiento − t_detonación` debe ser corto.
   Como los vecinos están separados por **6.5 frames**, esta ventana sola deja
   2–3 candidatos, y el espacio desempata.

**Refinamiento por punto fijo** (cierra el círculo con `PARABOLAS.md`): con
origen + impacto + tiempo de vuelo la balística queda determinada → estimas la
altura real de la roca → **corriges el paralaje con `k` propio de cada traza** →
reasocias. Converge en 2–3 vueltas.

**Dos advertencias para no equivocar la implementación:**
- **No es asignación 1-a-1.** Un pozo lanza varias rocas → min-cost flow **con
  capacidad**, no Hungarian puro. Conecta con §P7.
- **El eslabón débil es cuándo "nace" una traza**: el destello y el humo tapan el
  origen. Ese retardo hay que **medirlo**, no asumirlo — y se mide justamente con
  los trazos manuales de la Fase A cuyo pozo ya validamos. **La Fase A calibra la
  Fase B.**

---

## FASE C — E8: Exportación y medición

Explícitamente **después** (decisión del usuario: "por ahora la parte visual es
lo importante"). Queda anotado para no perderlo:

- Distancia de alcance por roca, en metros. **Solo origen e impacto son
  convertibles con `h_matrix`** — están en `z ≈ 0`. Los puntos intermedios del
  arco están en el aire y `H⁻¹` los manda a una posición equivocada
  (`PARABOLAS.md` §6).
- El CSV trae **cota `Z` por pozo** (~3175.8 m): sirve para distancias reales
  cuando haga falta.
- Reporte por pozo: cuántas rocas, alcance máximo, dirección predominante.
- Envolvente de alcance → zonas de riesgo.

---

## Estado al cierre del 2026-08-06

**FASE A completa (E1–E4b).** Demo funcional, punto de recuperación etiquetado en
git. Falta la Fase B (tiempo) y la Fase C (medición).

| Archivo | Qué es |
|---|---|
| `demo/trayectorias.html` | El visor. **Un solo archivo**, malla embebida, se abre con doble clic |
| `malla_export.py` | Proyecta los 113 pozos con la `h_matrix` ya calibrada y los **inyecta** en el HTML. Correr si se recalibra |
| `frame_export.py` | Saca un frame del video original por índice **del clip** (guarda el offset 12.5 s, que estaba solo en un comentario) |
| `test_asociacion.py` | Banco de pruebas del algoritmo contra trayectorias sintéticas con origen conocido |
| `PARABOLAS.md` | Por qué la parábola no se ve parábola desde el dron |

**Insumos para abrir el visor:** máscara `out/6_mascaras/1_intensidad.png` y frame
`out/4_fase0_referencias/frame_clip_045.png` (terreno seco, 3 frames antes de la
primera detonación — es donde mejor se juzga el calce de la malla).

**Lo que NO está verificado:** la interacción del visor solo se probó por
sintaxis y por renders independientes en Python; el algoritmo está medido contra
trayectorias **sintéticas**, no contra trazos reales con origen conocido.

---

## Resumen de decisiones tomadas

| # | Decisión | Motivo |
|---|---|---|
| 1 | Fase A **solo con trazos manuales** | Error de detección cero → aísla la matemática de asociación |
| 2 | Salida por **candidatos**, con degradación a grupo y sector | Más defendible que un pozo único que el cliente puede desmentir |
| 3 | **Sin `.SRT`**; parámetro `k` con slider | La altura del DJI es relativa al despegue: en un rajo, miente |
| 4 | Cargar `h_matrix.json` + ajuste fino, **no** rehacer el ajuste | `pre_tiros.py` ya lo resuelve con RMS 3.19 px |
| 5 | Tolerancia angular **proporcional a la distancia** | Medido: 3° a 100 m = 1 pozo; a 200 m = 1.9 pozos |

## Lo que hay que decirle al cliente

Hizo **dos** reclamos y esta demo resuelve uno solo. Conviene separarlos:

1. *"Los trazos los debe detectar el algoritmo"* → es detección/tracking, **no**
   lo resuelve esta demo. Pero tenemos algo mejor que una promesa: sabemos **por
   qué** el algoritmo pierde las trayectorias largas — el retroceso por paralaje
   las parte en dos — y está derivado con fórmula en `PARABOLAS.md` §3.
2. *"No hay asociación al tiro de origen"* → esto sí, y es lo que se muestra.

El encuadre correcto de la demo no es *"yo dibujo y el sistema adivina"*, sino
**"validamos que la capa de asociación funciona, con trazos perfectos, antes de
conectarla a la detección"**. Es la estrategia human-in-the-loop ya acordada
(`ESTADO_Y_PENDIENTES.md` §3), no un cambio de plan.
