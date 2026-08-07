# Pendientes — Flyrocks

> **Índice de temas abiertos.** Este documento es el *menú*: una línea por tema
> para poder elegir cuál retomar. El detalle vive en los docs enlazados.
> Última revisión: 2026-08-05.
>
> Convención: cada tema tiene **estado**, **por qué importa** y **dónde está el
> detalle**. Al cerrar uno, moverlo a "Cerrados" con la fecha.

---

## Activos (lo que está en la mesa ahora)

### P0 — Asociación trayectoria → tiro de origen ⬅ **prioridad, pedido del cliente**
**Estado:** **FASE A (E1–E4b) IMPLEMENTADA Y ENTREGADA** el 2026-08-06.
Punto de recuperación en git: tag `demo-asociacion-v1`.
**Siguiente:** Fase B — sumar las detecciones del pipeline y el calce temporal.

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
**Estado:** propuesto, no implementado. **Pedido por:** equipo de backend.
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
