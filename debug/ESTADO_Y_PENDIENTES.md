# Estado del trabajo y pendientes — Flyrocks (denoising de trayectorias)

> Documento de traspaso para continuar en un chat nuevo. **Es el detalle
> narrativo** (qué se intentó y por qué falló). Para el *listado* de temas
> abiertos con IDs, ver **`debug/PENDIENTES.md`**. Para la geometría de por qué
> las parábolas no se ven parábolas desde el dron, ver **`debug/PARABOLAS.md`**.
> Última revisión: 2026-07-26.
> Entregable #1 = **máscara visual limpia de trayectorias de roca, SIN humo/polvo**.
> Los datos por roca (distancias, zonas peligrosas) vienen después.
> Estrategia acordada: **human-in-the-loop / semi-automático** (no 100% automático);
> tarea crítica de seguridad, bajo volumen, alto valor. Todo **por etapas con
> imágenes intermedias** para revisar. Ver detalle en `debug/PLAN.md`.

---

## 1. El problema en una línea

Sobre video cenital (dron) de una tronadura, separar **roca eyectada** (líneas
casi radiales/parabólicas) del **humo/polvo** (ruido persistente), y dibujar
trayectorias limpias. Los dos dolores: (a) **trayectorias falsas** por humo,
(b) **trayectorias fragmentadas** que no se conectan.

---

## 2. Pipeline actual que SÍ funciona (Camino 2)

1. **Estabilización** de la deriva del dron (optical flow + afín, RANSAC).
2. **Extracción de eventos** por diferencia de intensidad entre frames
   (`absdiff`, `noise_threshold=8`). **SIN filtro de energía/percentil**.
3. **Exclusión del área de polvo** por mapa de persistencia.
4. **Tracking de velocidad constante** con continuidad física (gate elíptico,
   sin teletransportes ni reversas) — reemplazó al Kalman+Hungarian original.
5. **Filtro temporal** (descartar trazas nacidas antes del inicio de tronadura).
6. **Filtro de secuencia** (traza-back a un tiro ya detonado).

Resultado sobre `clip_full`: **44408 → 8207 trazas "flyrock"**. Guardado en
`polar_result.npz` (traj [id,x,y,t] + origin + meta).

Insumos del usuario ya disponibles y validados:
- **CSV de secuencia** (`Secuencia (2).csv`): 129 tiros, `DetonatingTime` en ms
  (3000–5769). Inicio = **frame 48**. Propaga derecha→izquierda.
- **Homografía** (`getSafeAffineMatrix`, replicada en `debug/homografia.py`):
  valida ~8.5 px. `h_matrix` mundo→píxel guardada en `_cache/homografia.npz`.
- **PNG pintado** (verde/rojo/azul) sobre la máscara gris de estelas.

---

## 3. Semántica de la pintura (clave, definida por el cliente)

Los colores son **CLASES (intención), no identidades**. Un color por clase basta;
los cruces no importan.
- 🟢 **Verde = ignorar** (humo / no-interés).
- 🔴 **Rojo = zona con flyrocks** → **anula la exclusión de polvo** ahí (las rocas
  salen desde DENTRO del humo, no solo del borde).
- 🔵 **Azul = crítico / UNA sola trayectoria.** Un trazo azul = exactamente **una**
  roca. Relaja filtros para garantizar captura de las importantes (largo alcance
  y **curvas/parábolas**, que los filtros automáticos tienden a botar). Sirve de
  checklist de validación y de **mecanismo de rescate**: si falta una, la pinto y
  el algoritmo la reconstruye desde el set sin dedup.

ROI efectiva del tracker: `keep = blue | (red & ~green & (blast==0))`.

---

## 4. Qué se INTENTÓ y qué pasó (para no repetir)

### Filtros que MATABAN rocas reales (descartados)
| Intento | Por qué falló | Estado |
|---|---|---|
| **Percentil de energía 96–98** | Botaba las rocas tenues/lejanas (la mayoría). | ❌ Eliminado |
| **`max_dist=30`** en el tracker | Mataba rocas rápidas (saltan >30 px/frame). | ❌ Reemplazado por tracker CV con gate elíptico |
| **Filtro de velocidad** (rápidas=roca) | Eliminaba casi todas las **parábolas** (a más velocidad, menos parábola). | ❌ Revertido |
| **Ventana temporal corta** | Cortaba el descenso de las parábolas (el corte estaba al FINAL). Hay que procesar el vuelo completo (~12.5–27.6 s). | ❌ Corregido a ventana completa |

### Discriminadores roca/polvo automáticos que se estrellaron
Se probó separar roca de polvo por **velocidad, suavidad, persistencia y borde de
ataque** (`exp_temporal_leading.py`: lag = t − primera_llegada del píxel).
**Todos fallan** porque las propiedades de roca y polvo **se traslapan**. Conclusión
dura: **la discriminación 100% automática choca contra un muro**; por eso el
enfoque es semi-automático (priors del usuario).

### Dedup (colapsar duplicados)
- **Dedup angular por sector** (`clean_and_stitch.dedup`, `da=0.03`): funciona pero
  es **agresivo** → colapsa parábolas del mismo sector y **pierde trayectorias**.
  Da las **174** trayectorias de `1_dedup`. Rango útil observado: **174–2786**
  (banda recall/precisión). `da` chico = más trazas, `da` grande = menos.
- **Dedup v3 con tiempo + azul** (`exp_dedup_v3.py`): idea = mismo sector +
  tiempos disjuntos = rocas distintas (conserva parábolas). Muchos bugs cazados
  por el usuario:
  - `max(members, key=extent)` elegía la recta larga que cruza y **borraba la
    curva azul** del usuario.
  - `MIN_FRAC=0.5` con "toca" absorbía 4672/8207 trazas en 6 trazos.
  - Encadenar fragmentos por tiempo hacía **abanicos caóticos**.
  - `used.update(members)` **descarta silenciosamente los no-inliers** (borra los
    2 pedazos que el usuario quería). **← BUG AÚN PRESENTE.**
- **RANSAC de curva dominante** (`ransac_dominant` en `exp_dedup_v3.py`): sí
  encuentra curvas (25 en total; azul#5=8), pero arrastra el bug de arriba.

---

## 5. Hallazgo más reciente (el que dejó el hilo abierto)

Ejercicio `exp_azul_contenidas.py`: por cada trazo azul, cuántas trayectorias de
`1_dedup` van **mayoritariamente dentro** (≥50 % de su largo), no solo lo rozan.
El discriminador correcto es la **fracción del largo dentro del azul** (no "tocar").

| Trazo azul | Trayectorias mayoritariamente dentro |
|---|---|
| **azul#1** | **2** (67 % y 97 %) ← **la parábola: el que va + el que vuelve** ✅ |
| azul#2 | **0** ← el dedup agresivo la **borró por completo** |
| azul#3 | 7 |
| azul#4 | 7 |
| azul#5 | **29** ← 6 trazos de la derecha **pegados** → se ven como uno |
| azul#6 | 15 |

**Lo que esto valida / revela:**
1. El modelo del usuario es correcto: su parábola (azul#1) = **exactamente 2
   pedazos**. La reconstrucción = **conectar esos 2**.
2. Conteos altos tienen causa: **trazos pegados** (`connectedComponents` los ve
   como uno → azul#5) y **trazos anchos cerca del centro** (todo converge en el
   blast) atrapan vecinos.
3. **azul#2 = 0** demuestra que **no se puede trabajar solo sobre `1_dedup`** en
   zonas azules: hay que ir a un **set más rico** (los 8207 o un dedup suave) para
   no perder lo que el dedup agresivo mató.

Imagen: `debug/out/6_mascaras/trayectorias/exp_azul_contenidas.png`.

---

## 6. Enfoque propuesto (a construir) para reconstruir el azul

Por cada trazo azul:
1. Tomar los pedazos **mayoritariamente dentro** (fracción del largo), desde un
   **set rico** (no el dedup agresivo).
2. **Conectar** los que forman un **camino coherente** (dirección/tiempo continuos
   — el "va" + el "vuelve" de la parábola) en **UNA** trayectoria.
3. Descartar el resto.

Requisitos prácticos detectados:
- **Separar trazos azules pegados** (o que el usuario los pinte separados y más
  finos) para que cada uno dé su trayectoria.
- Arreglar el **bug `used.update(members)`** que descarta no-inliers.

Ejercicio de laboratorio sugerido: construir el paso "**conectar**" sobre
**azul#1 (la parábola, 2 pedazos)** como caso limpio, y luego generalizar.

---

## 7. Preguntas abiertas para el usuario (arrastradas)

1. ~~En la parábola (azul#1), los 2 pedazos ¿se tocan o hay hueco?~~
   **RESPONDIDO (2026-07-26):** son la **ida y la vuelta** de la parábola, **no se
   cruzan**. El sistema **las detecta bien a las dos**; el problema es puramente
   que **no las une**. Para el usuario es evidente a ojo que son una sola roca.
   → El déficit está en la **asociación**, no en la detección.
2. El usuario dijo tener **ideas nuevas** — pueden cambiar el enfoque antes de
   codear. **← Punto de partida del chat nuevo.**

**ESTADO 2026-07-26: hilo azul EN PAUSA** por decisión del usuario. Se vuelve
aguas arriba, a **limpiar/refinar la ENTRADA** (ver §11).

---

## 8. Parking lot (futuro, alta prioridad)

- **Asociación GLOBAL / tracklets** (min-cost flow / MHT / grafos sobre TODAS las
  detecciones a la vez), en vez del tracking greedy frame-a-frame. Ataca de raíz
  la fragmentación/duplicación. Probablemente lo que hacía la solución del cliente
  (~6 h, offline). Plan del usuario: dejarlo corriendo horas.
  - **C1**: asociación a nivel de tracklets (~minutos–1 h en esta máquina).
  - **C2**: a nivel de detección (~6 h, en otra máquina).
- Flujo óptico **denso** para separar humo/roca por campo de velocidad (pesado,
  beneficio incierto).
- Clasificador ML por-trayectoria (requiere datos etiquetados).

---

## 9. Mapa de archivos (dónde está cada cosa)

**Librería central**
- `debug/clean_and_stitch.py` — `load_tracks()`, `dedup(tracks,origin,da)`
  (angular, keep-longest), `fit_and_extend()`, `render()`, `load_shots()`,
  `radial_angle()`, `apply_H()`.
- `debug/polar_v2.py` — tracker principal (Camino 2). `parse_paint()`,
  `build_persistence()`, tracker CV, `filter_tracks()` con relajación por azul.
  Flags `--paint`, `--no-video`, `--start-frame 48`. Escribe `polar_result.npz`.
- `debug/paint_filter.py` — `PaintMask` (`keep=(red|blue)&~green`, `in_blue()`),
  standalone para el pipeline del colega.
- `debug/homografia.py` — replica `getSafeAffineMatrix`; H en
  `_cache/homografia.npz`.

**Experimentos de azul (el hilo vivo)**
- `debug/exp_azul_contenidas.py` — **el más reciente y útil**: fracción-dentro por
  trazo azul. → `exp_azul_contenidas.png`.
- `debug/exp_overlay_azul.py` — azul semi-transparente sobre `1_dedup`; clasifica
  cuáles pasan por el azul (91 de 174).
- `debug/exp_dedup_v3.py` — dedup con tiempo+azul + RANSAC. **Tiene el bug
  `used.update(members)`.**

**Otros experimentos**
- `debug/exp_temporal_leading.py` — filtro de borde de ataque (lag). No discrimina
  (roca y polvo se traslapan).

**Salidas ordenadas**
- `debug/out/6_mascaras/` — máscaras (1_intensidad, temporal, combinada) y
  `trayectorias/` (imágenes de los experimentos).
- `debug/out/4_fase0_referencias/` — lienzo para pintar
  (`lienzo_para_pintar_gris - Copy.png`).
- `debug/out/_cache/` — cachés (first_arrival.npy, homografia.npz).

**Docs**
- `debug/PLAN.md` — plan estratégico vivo (fases, catálogo de inputs, semántica).
- `debug/ESTADO_Y_PENDIENTES.md` — **este documento** (qué se intentó + pendientes).
- `debug/PREPROCESO.md` — **etapa activa desde 2026-07-26**: refinamiento de la
  entrada (modelo de fondo del pre-roll, realce de linealidad, binarización con
  histéresis). Ver §11.

---

## 11. Giro de enfoque (2026-07-26): refinar la ENTRADA

El usuario decide **pausar el hilo azul** y volver aguas arriba. Diagnóstico
acordado: la información no sobra, **está en el eje equivocado** — intensidad,
velocidad y persistencia ya demostraron traslaparse (§4); la **forma local**
(línea vs mancha) no.

Etapa acotada a **generar máscaras e imágenes de preprocesamiento**, sin tocar el
tracker ni reescribir código de experimentos anteriores. Salidas en
`debug/out/7_preproceso/`. **Detalle completo, decisiones y orden de trabajo en
`debug/PREPROCESO.md`.**

---

## 10. Cómo retomar en el chat nuevo (arranque rápido)

1. Leer `debug/PLAN.md` + este documento.
2. El hilo abierto es **§6**: reconstruir la trayectoria azul conectando los
   pedazos mayoritariamente-dentro, empezando por **azul#1** (parábola, 2 pedazos).
3. Antes de codear: escuchar las **ideas nuevas** del usuario (§7.2) y resolver si
   los 2 pedazos se tocan o hay hueco (§7.1).
4. Recordar las restricciones aprendidas: **no** filtros de energía/velocidad,
   ventana **completa**, dedup agresivo **pierde** cosas → usar set rico en zonas
   azules, y **no** promediar/encadenar a la fuerza (hace abanicos falsos).
