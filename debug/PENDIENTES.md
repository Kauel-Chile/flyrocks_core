# Pendientes — Flyrocks

> **Índice de temas abiertos.** Este documento es el *menú*: una línea por tema
> para poder elegir cuál retomar. El detalle vive en los docs enlazados.
> Última revisión: 2026-08-09.
>
> Convención: cada tema tiene **estado**, **por qué importa** y **dónde está el
> detalle**. Al cerrar uno, moverlo a "Cerrados" con la fecha.

---

## Activos (lo que está en la mesa ahora)

### P11 — Vista prototipo sobre caso congelado ⬅ **en construcción, 2026-08-09**
**Estado:** caso congelado ✅ · visor v1 escrito, **sin probar en navegador**.

El objetivo es iterar la vista final **sin depender del frontend ni del
pipeline**. Dos piezas:

1. **`caso_export.py`** — corre el pipeline una vez y congela todo en
   `debug/casos/<nombre>/` (`caso.json` + clip + frame + máscara). Idempotente
   por etapas. Las zonas se **derivan de la malla** (hull de los 113 pozos,
   +80 m para seguridad), no se dibujan: así el caso es reproducible.
2. **`demo/vista.html`** — el visor. Se sirve con `caso_serve.py` (con
   `file://` el navegador bloquea el fetch del caso).

**Caso `3160-789` medido:** 34.9 M eventos → 3.041 trayectorias → **763 rocas**
(588 Proyección, 168 peligrosa, 7 fuera de vista). Pipeline 92 s, de los cuales
**60 s son el GridSearch** (justo el nodo que P1 quiere cachear).

**Decisión de diseño — un estado en vez de tres capas.** La vista del wizard
esconde trayectorias por tres mecanismos independientes (3 sliders, checkboxes
de clase, capa de borrados) y cuando algo desaparece no hay forma de saber cuál
lo escondió. Acá hay un `estado` (activa/oculta/descartada) + `razon`, y el
panel muestra el desglose. El descarte manual gana sobre los filtros.

**Hallazgo — los rangos de los sliders del front están mal calibrados:**

| Filtro | Slider front | Datos reales | Efecto |
|---|---|---|---|
| tortuosidad | 0–5 | 1.00–2.18 (med 1.04) | usa el 4% del recorrido; en 5.0 no filtra nada |
| escape_relativo | 0–10 | 0–3.34 (med 0.70) | un tercio del rango |
| r2_score | 0–1 | 0–1 (med 0.96) | bien |

En nuestra vista los rangos se **derivan del caso**. Además el front compara
`Math.log(r2) >= Math.log(r2Score)`, que es equivalente a comparar directo y
solo funciona con el 0 por accidente (`-Inf >= -Inf`).

**Ojo al asociar:** mediana de **12 puntos** por trayectoria pero mínimo **2**.
Con 2 puntos la tangente inicial no es confiable — hay que exigir un largo
mínimo y decirlo, no asociar a ciegas.

**Bug del core a reportar:** los `print` con emoji de `trajectory_filters.py:48`
revientan el pipeline entero con `UnicodeEncodeError` en consola Windows sin
UTF-8 (mató una corrida en el nodo 12, tras 98 s). A ellos no les pasa porque
corren en Docker.

**Siguiente:** probar el visor en navegador, enganchar la asociación (P0) y
recién después los entregables nuevos (heatmap, proyección de tiros).

---

### P0 — Asociación trayectoria → tiro de origen ⬅ **prioridad, pedido del cliente**
**Estado:** Fase A entregada el 2026-08-06 (tag `demo-asociacion-v1`).
**FASE B — CALCE TEMPORAL: implementada el 2026-08-09** sobre trayectorias
reales del pipeline, en `demo/vista.html`.

> **El nombre es "calce temporal"** (Fase B, E5–E7 del plan). Es cruzar el
> **nacimiento** de cada traza con el **tiempo de detonación** de cada pozo.

**El eje temporal existía y se estaba tirando.** El tensor del tracker es
`[id, x, y, t]` y `HighVelocityFilterNode` lo ordena por `(id, t)` ascendente,
pero `trajectory_categorization.py` exportaba solo `[:, 1:3]`. Se agregó
`"frames"` al JSON (cambio aditivo, no rompe nada) → **hay que avisarle al
equipo del core**.

Verificado de paso: `puntos[0]` **sí** es el origen temporal (está más cerca de
un pozo en el 75% de los casos, 28 m contra 43 m). No hay inversión.

**Lo que aporta el tiempo, medido sobre las 763 trayectorias reales:**

| | origen | probable | zona | sin candidato |
|---|---|---|---|---|
| Solo geometría | 231 | 91 | **261** | 58 |
| Con calce temporal | **352** | 128 | **92** | 69 |

Las de baja confianza caen **65%**; 276 trayectorias cambian de pozo. El
**16.3%** de las asociaciones puramente geométricas **violan causalidad** (el
pozo elegido detonó después de que la traza ya era visible): son imposibles, y
el tiempo las caza sin ningún parámetro que calibrar.

**Diseño:** el término temporal entra **dentro** del ranking, no como filtro
posterior — así una trayectoria cuyo mejor candidato geométrico es imposible se
reasigna al siguiente compatible en vez de quedarse huérfana.

**El retardo de nacimiento está medido y NO es limpio.** `dt` = nacimiento −
detonación, sobre las asociaciones de conf ≥ 0.55: p25 = **3 frames**, mediana
= **50**, p90 = **192**. O sea hay un grupo que aparece casi de inmediato y una
cola larguísima. Coherente con lo previsto: **el destello y el humo tapan el
origen**. Por eso la ventana es un slider y no una constante.

**Siguiente:** calibrar `k` con el ojo del usuario sobre el terreno (sigue en
0 = paralaje apagado), y el refinamiento por punto fijo (E7): con tiempo de
vuelo → altura real → `k` por trayectoria en vez de global.

El cliente vio el aplicativo y dijo que no le sirve: los trazos no dicen **de qué
pozo salió** la roca. Su versión cruza malla de detonación + tiempos de secuencia
para asociar por posición, velocidad y tiempo.

Plan en **8 etapas**: E1–E4 = demo 2D sobre trazos dibujados a mano (malla +
paralaje + cuña de retroceso + candidatos); E5–E7 = calce temporal sobre
detecciones; E8 = exportación y distancias.

Hallazgo que gobierna el diseño: **geometría y tiempo son complementarios**. Los
pozos que el espacio no distingue (5.45 m = 47 px) detonan separados por
**216 ms = 6.5 frames**; los que el tiempo no distingue (<1 frame) están a
19.7 m. Por eso la fase 2D solo puede dar **candidatos**, no el pozo.

Decisión: **no se usará el `.SRT` del dron** (su altura es relativa al punto de
despegue; en un rajo con el operador en otra cota, miente). La corrección de
paralaje se hace con un escalar `k = z/(h−z)` calibrado por slider.

Detalle completo: **`debug/PLAN_ASOCIACION.md`**.

---

### P1 — Refactor de iteración del pipeline (caché + contratos + CLI)
**Estado:** **CACHÉ POR NODO IMPLEMENTADA Y MEDIDA (2026-08-11).** Faltan los
contratos declarados y el CLI.

**Cómo funciona:** cada nodo tiene una llave encadenada
`hash(clase + parámetros + llave del nodo anterior)`, y su salida se guarda bajo
esa llave (`utils/nodes/base.py`). Si una entrada cambia, la llave cambia y con
ella la de todos los nodos siguientes. **No hay invalidación manual ni flags:**
la corrección sale de la identidad. Se apaga con `PIPELINE_CACHE=0`.

**Medido sobre el caso 3160-789:**

| Escenario | Antes | Ahora |
|---|---|---|
| Corrida limpia | 92 s | 104 s (+13 %, escribir la caché) |
| Repetir sin cambios | 92 s | **0 s** |
| Cambiar `sigma` (nodo 8) | 92 s | **5 s** |
| Cambiar `esp` (nodo 3) | 92 s | 75 s |

El último es correcto: el GridSearch (nodo 4, 58 s) depende del clustering
(nodo 3), así que tiene que recalcularse. La caché no inventa atajos.

**Dos decisiones de implementación:**
- **El nodo 1 no se cachea** (`EventExtractorNode.cacheable = False`): produce el
  tensor crudo de 35 M eventos (~1 GB). No se pierde nada — el nodo 2 lo filtra
  a 1.4 M y su caché ya trae todo lo necesario, así que al reanudar el nodo 1 ni
  se ejecuta.
- **Almacenamiento deduplicado por contenido.** Cada valor del contexto se
  escribe una vez bajo el hash de sus bytes; la entrada del nodo es un índice
  `{clave: hash}`. Sin esto cada nodo guardaba el contexto entero y el mismo
  tensor de 45 MB quedaba escrito doce veces: **659 MB medidos por corrida,
  contra 60 MB ahora**.

**Pendiente:** la caché **crece sin límite** — cada configuración distinta suma
objetos (74 MB tras tres). Falta una purga por antigüedad o por tamaño máximo.
No urge en desarrollo, sí antes de que esto llegue a un cliente.

`caso_export.py` acepta ahora `PERCENTILE`, `SIGMA` y `ESP` por variable de
entorno, para iterar sin editar el archivo.

---

### P17 — La vista como último paso intercambiable ✅ **base hecha 2026-08-11**

La idea (del usuario): que el último paso del wizard sea una **bifurcación** —
nuestra vista o el Step4 del colega— para no tener que decidir hoy cuál gana. Es
un contrato de etapa con dos implementaciones; funciona si ambas consumen y
producen lo mismo.

**Lo que faltaba y ya está:** el core recibía `h_matrix`, `origin_zone` y
`expected_projection_zone` como form-data, los usaba y **los tiraba**. Terminado
el análisis nadie podía reconstruir con qué se hizo — solo el navegador, en
memoria. Ahora se guardan en `Job.entrada` (un JSON, no seis columnas, para
poder sumar campos sin migrar) y salen por `GET /api/results/{job_id}`.

**La vista tiene dos fuentes**, con el mismo comportamiento:

    vista.html?caso=3160-789          caso congelado en disco (iterar sin backend)
    vista.html?job=<id>&api=<url>     un análisis del core, por su job_id

**Verificado contra el core real:** los valores reconstruidos coinciden con los
del caso local — escala 8.5495 px/m, diámetro equivalente 60.29 m, radio de
evacuación deducido 101.1 m contra 100 reales, 726/726 con eje temporal.

**Falta el CSV de secuencia (importante).** Hoy lo lee el navegador en el paso 3
y nunca llega al backend, así que un job **no trae la malla de tiros** y sin
malla no hay asociación. La vista avisa en vez de fallar. Al resolverlo: subirlo
en el paso 3 y guardarlo en `Job.entrada`, o mandar los pozos ya proyectados.

**Migración de esquema:** `SQLModel.metadata.create_all()` solo crea tablas que
faltan, **no altera las existentes**. Al desplegar sobre una base ya creada el
core arrancaba bien y reventaba al guardar (`table job has no column named
entrada`). Se agregó `migrar()` en `database.py`, que corre en cada arranque y
hace `ALTER TABLE ADD COLUMN` de lo que falte. En SQLite es instantáneo.

---

### P16 — Persistencia del standalone ✅ **hecho 2026-08-11**

El `detovision.bat` del cliente corre `docker compose down` **al iniciar y al
detener** (y el arranque además usa `--build`). Sin volúmenes eso borraba las
dos SQLite y todos los archivos: **el cliente empezaba de cero en cada sesión**.

- Los datos se movieron a `/app/data` (antes estaban junto al código en `/app`,
  que no se puede montar sin tapar la aplicación). `DATA_DIR` es configurable.
- El `docker-compose.yml` monta `core_data` y `blast_data`, y se corrigió la
  ruta rota del blast detector (`./flyrocks` era su nombre hasta la v3).
- **El `.bat` no necesita cambios:** `docker compose down` no borra volúmenes
  nombrados; solo lo haría `down -v`, que el script no usa.
- La retención del limpiador pasó de 2 h fijas a **`RETENCION_HORAS`, 24 h por
  defecto** — lo que empiezas hoy sigue mañana. El equipo interno la sube por
  entorno sin tocar código.

**Ojo:** el compose canónico del proyecto no está claro. El de la raíz del
workspace estaba desactualizado y el que usa el cliente vive dentro del paquete
`Detovision_V3/Detovision/mvp/`. Hay que decidir cuál manda y sincronizarlos.

---

### P1 (contexto original) — lo que sigue pendiente
**Pedido por:** equipo de backend.
**Solo para desarrollo**, la entrega al cliente va en JavaScript igual.

El problema: `src/utils/services.py:76` corre los 11 nodos **siempre**. Para tocar
el nodo 5 hay que pagar el 1 (decodifica el video completo) y el 4 (GridSearch
multiproceso). Los parámetros están clavados en el constructor
(`services.py:40-50`), así que cambiar uno es editar el código. Y el contexto es
`Dict[str, Any]` sin contrato: si alguien cambia una salida, revienta tres nodos
después con un `None` incomprensible.

Los tres arreglos, en orden de impacto:
1. **Caché por nodo** (~40 líneas en `nodes/base.py`) — llave = hash de
   (clase + params + llave del nodo anterior). Cambias un parámetro del nodo 5 y
   solo se recalcula del 5 en adelante, sin flags. **Es el 90% del beneficio.**
2. **Contratos declarados** (~15 líneas) — cada nodo declara `INPUTS`/`OUTPUTS` y
   la cadena valida en la frontera. Migrar a Pydantic después (ya es dependencia
   vía FastAPI + SQLModel).
3. **CLI `--desde/--hasta` + `caso.json`** — congela video, zonas y `h_matrix`
   para no volver a dibujarlas en cada vuelta. Mismo patrón que ya funciona en
   `pre_tiros.py` → `h_matrix.json`.

**Recomendación dada: NO usar Tkinter.** No resuelve ninguno de los tres dolores,
no corre en el Docker existente y es trabajo desechable (la entrega es JS). Si se
quiere inspector visual, hacerlo web sobre el FastAPI que ya está montado: sirve
los artefactos cacheados y de paso es prototipo de la entrega.

**Bug a arreglar de paso:** `EnergyPercentileFilterNode` sobrescribe su propia
entrada (`trajectory_analysis.py:147`, `context["tensor_raw"] = filtered_tensor`).
Después de ese nodo el nombre de la clave miente. Renombrar la cadena
`tensor_raw` → `tensor_filtrado` → `detecciones`.

**Ofrecido:** implementar caché + contratos sobre `base.py` y dejar `services.py`
usándolos, para entregarlo funcionando y no como propuesta.

---

### P2 — Recalibrar y re-exportar en `pre_capas.py`
**Estado:** pendiente del usuario. **Bloquea a P3.**

La calibración exportada (`out/7_preproceso/06_capas/capas_params.json`) es la
**anterior** a las mediciones. Valores corregidos, ya medidos:

| Capa | Exportado | Corregido | Por qué |
|---|---|---|---|
| Linealidad | p80.4 | **p97–p98** | a p80.4 el 82% de la máscara es UN componente (derrame) |
| Intensidad | p90.8 | **p99** | a p90.8 pasan las 107 componentes → el filtro no filtra |
| Z-score | p86.0 | **descartar** | Jaccard 0.57 con intensidad: es redundante |
| anti-pelusa | 0 | **~2.0** | deja 68 trazos vs 39 pelusas, conserva parábolas |
| humo | — | **p95 / largo 558** | 91.1% cae dentro del verde pintado (azar: 15.8%) |

Resultado esperado con eso: 49/68 trazos, 0.79% del cuadro, 92.6% en zona de interés.

Detalle: `debug/PREPROCESO.md`.

---

### P3 — Puente máscara → tracker
**Estado:** decisión pendiente del usuario. **Es el cuello de botella real.**

La máscara se calcula sobre la imagen **acumulada**; el tracker consume **eventos
por frame**. No encajan. Dos opciones planteadas:
- **(a) ROI espacial** — la binaria como máscara de "dónde mirar". Simple, ataca
  el humo, no ayuda con la fragmentación.
- **(b) Acumulación en ventana corta** (5–10 frames) → tracklets orientados.
  Más trabajo, pero conserva la información de forma que es justo lo que hace
  valiosa a la linealidad.

Hasta que esto se resuelva, todo el trabajo de máscaras **no llega al resultado final**.

---

### P4 — Probar la máscara de humo sobre z-score
**Estado:** chico, listo para hacer. Tecla `N` en `pre_capas.py` cicla la fuente.

La pregunta: ¿atrapa una zona de humo distinta que la intensidad no ve? Si sí,
el z-score se salva de ser descartado en P2 (pero para *esto*, no como capa de trazos).

---

## En pausa (por decisión del usuario)

### P5 — Reconstrucción de trayectorias azules (parábolas)
**Estado:** pausado 2026-07-26 para trabajar aguas arriba en la entrada.

El hallazgo que dejó el hilo abierto: azul#1 = exactamente **2 pedazos** (la ida y
la vuelta de la parábola). El sistema **las detecta bien a las dos**; el déficit
está en la **asociación**, no en la detección. azul#2 = **0** pedazos: el dedup
agresivo la borró entera → no se puede trabajar solo sobre `1_dedup`.

**Bug conocido y aún presente:** `used.update(members)` en `exp_dedup_v3.py`
descarta silenciosamente los no-inliers (borra justo los pedazos que el usuario
quería).

**Novedad 2026-08-05:** `demo/trayectorias.html` exporta las trayectorias como
JSON (3 puntos de Bézier en coordenadas de imagen). Eso es un input **mucho
mejor que el PNG pintado a dedo** para este hilo: la parábola llega con su
geometría explícita, sin que el algoritmo tenga que adivinar la intención desde
un trazo grueso. Considerarlo al retomar.

**HIPÓTESIS FUERTE (2026-08-05) — posible causa raíz de los "2 pedazos":**
Simulando un flyrock balístico y proyectándolo por una cámara pinhole cenital
sale un criterio verificado en 42 combinaciones, **sin excepciones**:

> si `altura_dron < 4 × altura_máxima_de_la_roca`, la traza **se devuelve**
> en la imagen (se frena y retrocede hacia el centro)

No porque la roca vuelva: al caer se aleja de la cámara y se encoge más rápido
de lo que avanza. Con el dron a ~250 m, toda roca que suba de ~62 m lo hace.

**Por qué importa:** el tracker de `polar_v2` filtra por continuidad física
**sin reversas**. Si la traza real retrocede, ese filtro la parte en dos justo
en el punto de retroceso → daría **exactamente 2 pedazos**, que es lo que el
usuario describe para azul#1 ("la ida y la vuelta"). Encaja demasiado bien.

**Qué falta para confirmarlo:** (a) medir la altura real del dron — la de la
simulación se dedujo de la escala de la homografía (8.58 px/m → 448 m de
encuadre) más un FOV típico; (b) verificar en `polar_result.npz` si los dos
pedazos de azul#1 se tocan cerca del punto de retroceso previsto. Si se
confirma, la reconstrucción es **permitir la reversa cuando la traza está
lejos del nadir**, no un dedup más listo.

Guion de la simulación: rehacer con `scipy` (ajuste Bézier por alternancia con
parametrización monótona; el ajuste ingenuo con Powell da resultados falsos).

**Actualización 2026-08-06 — el criterio ahora tiene derivación cerrada.**
Proyectando la balística por una pinhole cenital sale `t* = √(2h/g)` para el
punto de retroceso, y la condición `h < 4·z_max` **exacta**: reproduce clavado el
resultado de las 42 simulaciones. Matiz nuevo: con la roca **lejos del nadir** el
retroceso ocurre **siempre** (tiende al ápice), sin importar la altura del dron
— o sea `h < 4·z_max` es la condición para que se devuelvan *todas*, incluida la
radial pura, que es el caso más difícil.

Detalle: **`debug/PARABOLAS.md`** (geometría y fórmulas) ·
`debug/ESTADO_Y_PENDIENTES.md` §5–§6 (hallazgo experimental).

---

## Backlog (buenas ideas, sin fecha)

### P6 — Ensemble: trayectorias por máscara y consenso
Idea del usuario: generar trayectorias desde **cada** máscara por separado y
comparar solapes a nivel de **trayectoria** (no de máscara), para filtrar por
consenso. Endorsado, pero **después** de que el tracker funcione con la máscara
combinada — linealidad deriva de intensidad, así que los errores están
correlacionados y el ensemble podría dar falsa confianza.

### P7 — Asociación global / tracklets (min-cost flow, MHT)
Alta prioridad de fondo. Ataca de raíz la fragmentación y la duplicación, en vez
del tracking greedy frame-a-frame. Probablemente lo que hacía la solución original
del cliente (~6 h offline). C1 = a nivel tracklets (minutos–1 h en esta máquina);
C2 = a nivel detección (~6 h, otra máquina).

### P8 — Match automático de tiros por destellos de detonación
Usar `DetonatingTime` del CSV para detectar el destello de cada tiro y obtener las
correspondencias sin clicks. Hoy se hace a mano en `pre_tiros.py` (RMS 3.19 px,
verificación independiente 8.0 px). Decisión del usuario: backlog.

### P9 — Deriva del dron al proyectar la malla en frames tardíos
Medido: frame 69 vs 38 = <1 px, pero **frame 452 vs 38 = 18.5 px**. La homografía
está anclada al frame 69. Si algún día se proyecta la malla sobre frames tardíos,
hay que compensar.

### P15 — Entregables ✅ **primera versión completa, 2026-08-09**

> **Vocabulario acordado** (para dejar de dar vueltas al nombrar las cosas):
> **el detector** = el pipeline de 13 nodos de `flyrocks_core/src` (video →
> trayectorias, lo mantiene el colega) · **la vista** = `demo/vista.html` ·
> **el wizard** = el frontend React de 5 pasos · **el caso** = la carpeta
> congelada de `debug/casos/`.

| # | Entregable | Dónde se genera |
|---|---|---|
| 1 | JSON de trayectorias | botón en la vista |
| 2 | Imagen de intensidad sola | `salidas.py` (copia de la máscara) |
| 3 | Imagen final con trayectorias y empalmes | botón «Imagen final (4K)» de la vista |
| 4 | Heatmap + histograma | `salidas.py` |
| 5 | PDF | `salidas.py` |

**Un solo JSON, no uno interno y otro para cliente** (decisión del usuario): dos
formatos divergen y a las pocas semanas nadie sabe cuál es la verdad. Lo esencial
va arriba en cada trayectoria y lo técnico anidado en `deteccion`. Incluye
**todas** las trayectorias con su `estado`, y el bloque `parametros` con `k`,
nadir, σ y el calce temporal — sin eso el JSON no es reproducible.

**La imagen final se exporta desde la vista, no desde Python**, reusando las
mismas funciones de dibujo con el contexto desviado a un canvas 4K. Un solo
renderizador para pantalla y exportación, o se desincronizan a la semana.

**El heatmap es relativo y sin escala numerada** (pedido del usuario, y es lo
correcto): la pregunta es de dónde salió más material, no cuántas rocas. Así
queda inmune a los duplicados de [[P13]] y a que el detector siga mejorando —
si mañana encuentra 30% más trayectorias, un mapa numerado cambia entero y uno
relativo conserva la forma. Sin número impreso no hay número que el cliente
pueda desmentir contando cráteres.
El peso va por **raíz** del conteo: en lineal, el pozo con 81 rocas (contra una
mediana de 3) saturaba la rampa y dejaba todo lo demás en negro.

**El histograma mide la distancia FUERA DEL ÁREA, no la recorrida.** Es la
corrección al enfoque de Yeriel, y la detectó el usuario: el radio de evacuación
se mide desde la voladura, no desde donde nació la roca. Una que nace en el
borde y viaja 80 m sale del área; otra que nace en el centro y viaja lo mismo se
queda dentro. Se usa `escape_relativo × diámetro_equivalente`, que ya calcula el
detector. Las «Fuera de vista» van con textura: su distancia está **censurada**
(es un mínimo, la roca siguió fuera del cuadro).

**Medido en el caso 3160-789:** 352 con tiro identificado, 128 grupo, 92 sector ·
80 tiros con material · **109 de 763 rocas sobre el radio de evacuación** de
100 m · alcance máximo 249 m, mediana 10 m.

**Deuda conocida:** para poder probar `salidas.py` sin navegador se generó el
entregable con un script Node que **replica** el asociador de la vista. Son dos
implementaciones del mismo algoritmo y van a divergir. Al retomar: extraer el
asociador a un `demo/asociacion.js` que la vista cargue con `<script src>` y el
CLI pueda importar.

---

### P14 — Pendientes de la vista (no bloquean los entregables)
**Estado:** anotados el 2026-08-09 al pasar a trabajar en las salidas. Revisar
después; ninguno contamina un entregable.

- **Unir trayectorias partidas por sus extremos.** La herramienta ya existe en
  el frontend del colega (`saveNewProjection()`), acá no. Es el parche manual a
  la fragmentación de [[P5]] (el retroceso por paralaje parte los vuelos largos).
- **Tortuosidad con escala logarítmica sobre `tortuosidad − 1`.** El control
  invertido ya va en la dirección correcta, pero el último tramo cae de 571 a
  166 trayectorias de golpe: hay una masa enorme con tortuosidad exactamente
  1.000 (los trazos cortos de 2–3 puntos, perfectamente rectos).
- **Rendimiento del zoom sobre la máscara 4K.** No medido en uso real; si va a
  tirones, separar la capa de fondo en su propio canvas cacheado.
- **Nadir: sigue en el centro del cuadro.** Solo importa cuando `k > 0`, así que
  el orden correcto es calibrar `k` primero y ajustar el nadir después.

**Bloqueante real para los entregables (no es de la vista):** `k` sigue en 0, o
sea el paralaje apagado. Cualquier salida generada ahora lleva sesgo sistemático
hacia afuera en el origen. Lo tiene que calibrar el usuario con el ojo sobre el
terreno, rango físico 0.02–0.11.

---

### P13 — Trayectorias duplicadas casi superpuestas
**Estado:** detectado por el usuario el 2026-08-09 al editar el caso. **Diferido**
a propósito: viene del pipeline, no de la vista, y no toca arreglarlo en esta etapa.

Hay proyecciones tan pegadas que **ni con zoom se distinguen** (pocos píxeles de
separación). El efecto práctico al depurar a mano es desmoralizante: borras una
creyendo que quedó limpio y detrás aparecen tres más.

Es el mismo objeto detectado varias veces por el tracker — la cara opuesta de la
fragmentación de [[P5]]: allá una trayectoria se parte en dos, acá una roca
genera varias trayectorias casi idénticas. **El origen es el tracking greedy
frame a frame**, que es exactamente lo que P7 (min-cost flow / MHT) ataca de
raíz.

Dos caminos cuando se retome:
- **Paliativo en la vista:** fusionar automáticamente las que compartan
  trayectoria dentro de una tolerancia, y mostrarlas como una con un contador
  ("×3"). Barato, ataca el síntoma, evita el borrado en cascada.
- **De fondo:** P7. Es el arreglo correcto pero es trabajo mayor.

Nota: el frontend del colega ya tiene la herramienta de **unir dos proyecciones
por sus extremos**, que es el parche manual a este mismo problema.

### P12 — Área de seguridad: igualar la del frontend ✅ **resuelto 2026-08-09**

El front **no** usa turf para esto: usa `offsetConvexPolygonRounded` en
`src/utils/geometry.ts`, un offset convexo con arcos, llamado con
`arcSegments = 5`. Y lo hace **en metros sobre las coordenadas del CSV**,
proyectando a píxeles recién después.

`caso_export.py` ahora replica esa construcción exacta (`offset_redondeado`).
La zona pasó de 13 vértices angulosos a **90 con arcos**, y está a **80.0 m del
hull en todos sus puntos** (min = max, verificado).

El detalle que importaba: hacer el offset en **metros y luego proyectar**, no en
píxeles. Un disco en metros no es un disco en píxeles si la homografía no es
isotrópica.

**El valor también quedó sincronizado:** `polygonDistance` arranca en **100 m**
en `Step3.tsx:78` (tope 300), y ese mismo número viajaba al generador de PDF
como `radio_equipos`. `SEGURIDAD_M = 100` ahora. Es un default, así que si en
una tronadura el usuario lo cambia en la UI, hay que reflejarlo en el caso.

### P10 — Otros (baja prioridad, beneficio incierto)
- Flujo óptico **denso** para separar humo/roca por campo de velocidad (pesado).
- Clasificador ML por-trayectoria (requiere datos etiquetados que no existen).

---

## Cerrados

| Fecha | Tema |
|---|---|
| 2026-07-26 | `entrega/linealidad.py` autocontenido para el colega (bit-idéntico al original) |
| 2026-07-26 | `pre_tiros.py` — match de tiros + homografía con zoom (RMS 3.19 px) |
| 2026-07-26 | `pre_capas.py` — visor de capas superpuestas + combinación espacio/semillas |
| 2026-08-04 | Máscara de humo automática (idea del usuario): 91.1% de acierto vs el verde pintado |
| 2026-08-05 | `demo/trayectorias.html` — trazado de parábolas sobre la máscara + exportación (demo cliente) |
| 2026-08-06 | `debug/PARABOLAS.md` — por qué la parábola no se ve parábola en la imagen, y qué rol juega el trazado manual |
