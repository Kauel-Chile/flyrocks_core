# Refinamiento de la ENTRADA — máscaras de preprocesamiento

> Documento vivo. Abierto: **2026-07-26**.
> Origen: conversación en la que el usuario decide **pausar el hilo azul** y
> volver aguas arriba, a **limpiar la entrada** antes de seguir peleando con la
> reconstrucción de trayectorias.
> Complementa `debug/PLAN.md` (estrategia) y `debug/ESTADO_Y_PENDIENTES.md` (historia).

---

## 1. Por qué volvemos aguas arriba

Hipótesis del usuario: *"más info es mejor" puede estar equivocado; puede que
estemos complicando en vez de simplificar. El algoritmo de visión no ve como el
ojo humano — lo que yo veo claro, al algoritmo le cuesta.*

Diagnóstico acordado: **la información no sobra, está en el eje equivocado.**
Roca y polvo ya demostraron **traslaparse** en intensidad, velocidad, suavidad y
persistencia (ver `ESTADO_Y_PENDIENTES.md` §4). Seguir refinando el eje de la
**intensidad** es seguir midiendo donde las dos clases están mezcladas.

Esta etapa trabaja **solo en preprocesamiento**: generar máscaras e imágenes.
No se toca el tracker ni el código de experimentos anteriores.

---

## 2. Hechos técnicos verificados en el código (2026-07-26)

Importantes porque acotan qué se puede mejorar y dónde:

- **La detección es diff frame-a-frame:** `absdiff(prev_gray, curr_gray)` con
  `diff > 8` — umbral **global y fijo** (`debug/harness.py:109-110`).
  Consecuencia: mide **cambio**, no **presencia**. Un objeto tenue o de bajo
  desplazamiento entre frames consecutivos **se auto-cancela**.
- **Un solo umbral global no puede servir a todo el cuadro:** 8 es demasiado alto
  en zonas oscuras (pierde rocas tenues) y demasiado bajo en zonas texturadas
  (mete ruido). ← causa probable del *bug de recall* (`PLAN.md` §5, punto 5).
- **La máscara gris que se usa de lienzo es una VISTA, no el insumo del tracker:**
  es `np.maximum` acumulado de los diffs (`debug/mask_combined.py:55`). El tracker
  consume **eventos por frame**. Un umbral que se ve bien en el acumulado puede
  dejar los frames sueltos casi vacíos.
- **Escena semi-estática confirmada por el usuario:** en ~90 % de los casos solo
  hay deriva del dron y algún reajuste de foco. **Sin vehículos ni personas en
  movimiento** (es norma de tronadura). → el pre-roll es un modelo de fondo válido.
- **Pre-roll disponible:** detonación en **frame 48** del clip → ~48 frames previos.

---

## 3. Las tres ideas en juego

### Idea A — El pre-roll como modelo de fondo ("pantalla verde")
*(idea del usuario, refinada en la conversación)*

Usar los frames previos a la detonación para caracterizar el fondo.

- **Lo que NO logra:** eliminar el polvo. El humo no existe en el pre-roll, así
  que el modelo no lo conoce y no lo puede restar. Descartar esa expectativa.
- **Lo que SÍ desbloquea (dos cosas, ambas valiosas):**
  1. **Diff contra fondo en vez de contra el frame anterior.** Mide *presencia*
     en lugar de *cambio* → la roca aparece con su contraste completo contra el
     terreno y deja de auto-cancelarse. Ataca directo el bug de recall.
  2. **Umbral adaptativo por píxel.** Con media y **desviación estándar** del
     pre-roll, el corte pasa de `> 8` global a `> k·σ(x,y)` local: "k sigmas
     sobre el ruido *de este píxel*".
- **Riesgos a manejar:**
  - **El flash de la detonación altera el auto-exposure de la cámara** → salto
    global de intensidad que invalida el modelo. Requiere normalización de
    iluminación global por frame.
  - **Dentro del humo el fondo queda tapado** y el diff-contra-fondo se satura.
    → diff-contra-fondo sirve **fuera** del humo (donde están las rocas de largo
    alcance, las críticas); frame-a-frame sirve mejor **dentro**. Son
    complementarios, no rivales: posible fusión por zona.

### Idea B — Realce de linealidad (cambiar de eje)
*(aporte de Claude en la conversación; es la apuesta fuerte)*

Donde roca y polvo **no** se traslapan es en la **forma local**:
- roca = **segmento lineal fino y orientado**;
- polvo = **mancha difusa sin orientación coherente**.

Es lo que usa el ojo humano (buena continuación / Gestalt): no ve píxeles
brillantes, ve *una línea que continúa*. Hoy el algoritmo umbraliza píxel a píxel
sin ninguna noción de "esto parece una línea".

Implementación: **banco de filtros orientados** (kernels de línea en ~16
direcciones, quedarse con la respuesta máxima) o filtro tipo **Frangi/Hessiana**.
Produce una máscara donde el valor ya no es "cuánto cambió" sino **"cuánto se
parece a una línea"**. Ahí una traza **tenue pero lineal supera a una mancha
brillante** — justo lo que hoy no se logra.

- **Riesgo honesto:** el polvo a veces sale en *jets* filamentosos, también
  alargados; y cerca del blast todo converge y se apelmaza. No es magia. Pero es
  el primer eje probado donde las clases no están demostradamente traslapadas.

### Idea C — Máscara binaria con slider
*(idea del usuario, refinada en la conversación)*

- **Trampa identificada:** un **umbral global único** es exactamente el filtro de
  percentil 96–98 **ya descartado** por matar rocas tenues (`ESTADO_Y_PENDIENTES.md`
  §4). No repetirlo.
- **Versión correcta: histéresis de dos umbrales** (lógica de Canny). Umbral
  **alto** = semillas confiables; umbral **bajo** = *crecer* desde las semillas por
  conectividad. Una traza tenue que **toca** una semilla fuerte sobrevive
  **completa**; el ruido tenue **aislado** muere. Resuelve el dilema "si subo
  pierdo tenues, si bajo entra ruido".
- → La herramienta lleva **dos sliders, no uno**.

**Encadenamiento:** las tres ideas no son alternativas. A mejora el insumo, B
cambia el eje de medición, y C corta sobre el resultado. La idea C aplicada sobre
**linealidad** (B) en vez de sobre **intensidad** es un ejercicio distinto y mejor.

---

## 4. Decisiones tomadas (usuario, 2026-07-26)

1. **El slider corta sobre el ACUMULADO** (es donde se ven las estelas y donde el
   usuario puede juzgar). Agregado acordado: mostrar **al lado** cómo queda un
   **frame suelto** con ese mismo corte, para no llevarse la sorpresa de un
   acumulado bonito con frames vacíos.
2. **NO se pierde la máscara de grises original.** Toda binarización es una **capa
   adicional**, nunca un reemplazo. El gris se conserva como fuente; el binario es
   una vista parametrizable. Binarizar temprano y botar el gris destruye
   información irrecuperable.
3. **En esta etapa solo se generan máscaras e imágenes de preprocesamiento.**
   Nada de tocar el tracker ni reescribir código de pruebas anteriores.
4. **Salidas ordenadas en carpetas** para permitir feedback visual iterativo.
5. **Esto sirve para los dos consumidores:** el tracker propio y el algoritmo lento
   del cliente (~6 h). Un binario limpio de estelas es probablemente justo la
   interfaz que ese algoritmo consume.

---

## 5. Estructura de salidas

```
debug/out/7_preproceso/
  00_baseline/     — máscara gris actual congelada (referencia de comparación)
  01_fondo/        — media del pre-roll, mapa de sigma, diff-contra-fondo
  02_zscore/       — acumulado en sigmas (umbral adaptativo) vs baseline
  03_linealidad/   — realce orientado + mapa de orientación en falso color
  04_binaria/      — contact sheet de umbrales + salidas de histéresis
  _cache/          — modelo de fondo cacheado (media, sigma)
```

**Regla:** cada paso entrega al menos **una comparación lado a lado contra el
baseline**. Sin baseline no hay forma de saber si mejoramos.

**Scripts** (todos nuevos, prefijo `pre_`; no tocan el pipeline existente):

| Script | Qué hace |
|---|---|
| `pre_common.py` | rutas, registro, normalización de iluminación, helpers de visualización |
| `pre_fondo.py` | paso 2 — modelo de fondo del pre-roll (cachea `fondo.npz`) |
| `pre_pasada.py` | **etapa cara**: una sola lectura del video → 4 acumulados (`pasada.npz`) |
| `pre_render.py` | pasos 1, 3, 4 — renders y comparaciones (instantáneo, desde caché) |
| `pre_linealidad.py` | paso 5 — banco de filtros orientados (cachea `linealidad.npz`) |
| `pre_umbrales.py` | paso 6 — contact sheet + `hysteresis_guarded()` |
| `pre_slider.py` | paso 7 — herramienta interactiva (`--test` para correr sin ventana) |
| `entrega/linealidad.py` | **para entregar a terceros**: el paso 5 en un archivo único, sin dependencias del repo. Entra un PNG gris, sale la linealidad. Verificado bit a bit idéntico a `pre_linealidad.lineness()`. |
| `pre_capas.py` | paso 9 — visor de capas superpuestas + modo SOLAPE. Cruza cada región con la máscara pintada: es donde se midió que lo exclusivo de la linealidad es lo único enriquecido en zona de interés (74,4% vs 48,9% de línea base). |
| `pre_tiros.py` | paso 8 — match interactivo tiro↔píxel y matriz H. Reusa el ajuste afín de `homografia.py` (verificado idéntico); lo que aporta es obtener las correspondencias, que antes estaban escritas a mano en el código. |

Nota de implementación: cada frame se registra **directo contra el fondo** (no
encadenado), así no hay deriva acumulada a lo largo de los 15 s. Funcionó en los
453 frames sin un solo fallback, incluso con humo denso.

---

## 6. Orden de trabajo

| # | Paso | Entregable visual | Estado |
|---|------|-------------------|--------|
| 1 | **Baseline congelado** — regenerar la máscara gris actual tal cual | `00_baseline/` | ✅ |
| 2 | **Modelo de fondo del pre-roll** — media + mapa de σ | mapa de σ en falso color (muestra dónde el umbral 8 está mal calibrado) | ✅ |
| 3 | **Máscara z-score** — acumulado en sigmas, umbral adaptativo | lado a lado vs baseline: ¿aparecen las rocas tenues? | ✅ |
| 4 | **Diff-contra-fondo vs frame-a-frame** — los dos acumulados | lado a lado: efecto "pantalla verde" y dónde satura el humo | ✅ ❌ negativo |
| 5 | **Realce de linealidad** — banco orientado / Frangi | máscara de linealidad + mapa de orientación | ✅ **el hallazgo** |
| 6 | **Contact sheet de umbrales** — grilla de N binarizaciones | grilla, para acotar el rango antes de invertir en UI | ✅ |
| 7 | **Herramienta de doble slider** (histéresis) | UI con preview acumulado + frame suelto | ✅ |

**Nota de eficiencia:** el paso 5 (linealidad) es la apuesta más fuerte y también
la más incierta. Se puede adelantar un **vistazo rápido sobre el baseline** justo
después del paso 1, para saber temprano si promete, antes de invertir en los pasos
2–4. Decisión del usuario.

**Por qué la UI va al final:** los pasos 1–6 son baratos y reversibles; cuando
llegue el paso 7 ya sabremos sobre qué máscara y en qué rango vale la pena cortar.

---

## 6bis. RESULTADOS de la corrida 1→7 (2026-07-26)

### Paso 2 — el umbral fijo está 4× fuera de calibración (cuantificado)
Ruido medido en el pre-roll: `diff_sigma` **mediana 0.46**, p95 1.0 (muy bajo; el
blur 3×3 ya suaviza). Por lo tanto:

| El umbral fijo de 8, en sigmas locales | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| sigmas | 8.0 | 12.9 | **17.3** | 22.5 | 26.7 |

**98.2 % del cuadro** opera con un umbral > 6σ. Un corte estadístico normal
(4–5σ) equivaldría a **~2 niveles de intensidad**, no 8. Confirma numéricamente
el *bug de recall*: el pipeline actual es ~4× más estricto de lo necesario.
El fondo medio sale **nítido**, lo que valida la estabilización del pre-roll.

### Paso 4 — ❌ RESULTADO NEGATIVO: la "pantalla verde" no funciona
El diff-contra-fondo **se satura**: sobre 400 frames el humo termina barriendo
casi todo el cuadro, y el máximo por píxel refleja *el contraste del terreno
contra el humo*, no las rocas. La imagen resultante parece una foto del terreno.
**La idea A, en su forma "diff contra fondo sobre el acumulado", se descarta.**
Lo que SÍ sobrevive de la idea A es el **modelo de ruido** (paso 2), que es de
donde salió el diagnóstico del umbral.

### Paso 5 — ✅ EL HALLAZGO: la linealidad sí separa
El banco de filtros orientados **funciona**: las trayectorias saltan como líneas
nítidas y la veladura de humo se apaga. Confirma la hipótesis de cambiar de eje
(intensidad → forma).

Parámetros elegidos por barrido (`03_linealidad/variantes_escala.png`):
- **Largo del kernel = 51 px.** 15 px → domina la textura fina del terreno;
  81 px → el humo se "peina" en filamentos falsos (el riesgo anticipado se
  confirma); **51 px** → las estelas largas y las curvas dominan.
- **Supresión por DISCO, no por mediana.** `max_θ − mediana_θ` deja artefactos en
  estrella alrededor de puntos muy brillantes; `max_θ − respuesta_al_disco` los
  cancela (un punto responde fuerte al disco; una línea, débil).
- **Convolución, no apertura morfológica**: las estelas salen **punteadas** (la
  roca aparece intermitente frame a frame) y la convolución *integra* a lo largo
  de la dirección, uniendo los puntos; una apertura los rompe.

### Paso 6 — ⚠️ la histéresis pura SE DERRAMA (corrección a lo propuesto)
La recomendación original de histéresis simple **falla aquí**: en la zona del
blast la densidad conecta todo en un blob único.

| Método | % del cuadro | Componentes |
|---|---|---|
| umbral simple p99 | 1.00 % | 769 |
| histéresis pura p95/p99 | **4.20 %** | **83** ← pocas y enormes = derrame |
| **histéresis con salvaguarda** | 1.73 % | 624 |

**Salvaguarda** (`hysteresis_guarded`): por cada componente con semilla, si es
chica (`área ≤ max_area`) se acepta **entera** — la traza tenue aislada sobrevive
completa, que era el objetivo; si es un derrame (`área > max_area`) se conserva
solo su **núcleo fuerte**. Recupera área real de trazas tenues (1.00 → 1.73 %)
sin el blob. Valor usable: `max_area ≈ 4000`.

### Filtro de LARGO (agregado tras feedback del usuario, 2026-07-26)
Observación del usuario: en la zona de polvo del centro quedan **muchas líneas
cortas que son ruido de la explosión**. El realce de linealidad las enciende
igual que a una traza real porque *localmente sí son líneas*. Lo que las separa
no es el brillo ni la forma local: es el **largo**.

`length_filter()` descarta componentes cuya diagonal de bounding box sea menor a
`min_len`. Muy eficiente en este caso:

| largo mínimo | componentes | % del cuadro |
|---|---|---|
| sin filtro | 624 | 1.73 % |
| 40 px | 154 | 1.66 % |
| **80 px** | **98** | **1.48 %** ← mejor balance |
| 150 px | 54 | 1.13 % (ya se pierden trazas medianas) |
| 250 px | 24 | 0.71 % (demasiado agresivo) |

De 624 a 154 componentes **perdiendo solo 0.07 % de área**: lo eliminado era todo
diminuto. A 80 px la banda central queda limpia conservando las trayectorias.
Desde 150 px empieza a costar trazas reales. Expuesto como slider `largo_min px`.

### Paso 7 — herramienta lista
`debug/pre_slider.py`. Calibra sobre el acumulado con preview del frame suelto al
lado, y el preview **ya demostró su utilidad**: con lo=p95/hi=p99 el acumulado da
215 componentes (2.29 %) pero el frame #60 solo **7** (0.236 %) — porque está a 12
frames de la detonación. Exportar con `g` produce una **capa binaria adicional**
más un JSON de parámetros; la máscara gris se conserva intacta.

### ABLACIÓN (2026-07-27) — ¿los pasos 1–4 aportan? NO en este video
Pregunta del usuario: *¿puedo aplicar el paso 5 sobre la máscara de intensidad
que ya teníamos?* Se midió con `debug/pre_ablacion.py`, comparando dos acumulados
que difieren solo en dos factores (marco fijo + normalización de iluminación):

| | correlación | contraste de la linealidad (p99/mediana) |
|---|---|---|
| acumulado ANTIGUO (marco móvil, sin normalizar) | — | **10.3** |
| acumulado NUEVO (marco fijo + normalizado) | 0.85 vs antiguo | 8.8 |

Medias globales casi idénticas (11.4716 vs 11.4703); localmente sí difieren
(21 % de píxeles con >5 niveles de diferencia), **pero las dos máscaras de
linealidad resultantes son visualmente indistinguibles**
(`03_linealidad/ablacion_linealidad.png`).

**Conclusión: el paso 5 se puede aplicar DIRECTAMENTE sobre la máscara de
intensidad que ya existía.** El marco fijo y la normalización de iluminación
resuelven problemas reales (deriva grande del dron, cambio de exposición por el
flash) que **este** video no tiene de forma significativa — la escena es
semi-estática, como había indicado el usuario. Son salvaguardas útiles para otras
tronaduras, no un requisito del realce de linealidad.

Lo único de los pasos 1–4 que sí rindió es el **modelo de ruido** (paso 2), pero
como *diagnóstico* (el umbral fijo = 17σ), no como insumo del pipeline.

### Conclusión operativa
El orden de valor quedó: **linealidad (paso 5) >> umbral adaptativo (pasos 2–3) >
diff-contra-fondo (paso 4, descartado)**. La cadena recomendada es
`intensidad → linealidad L=51 → histéresis con salvaguarda`.

---

## 7. En pausa (retomar después de esta etapa)

El hilo azul (`ESTADO_Y_PENDIENTES.md` §6): reconstruir las trayectorias azules
conectando los pedazos mayoritariamente-dentro.

**Dato del usuario que lo aclara (2026-07-26):** en azul#1 las dos piezas son la
**ida y la vuelta** de la parábola, **no se cruzan**, y el sistema **las detecta
bien a las dos**. El problema es puramente que **no las une**. → el déficit está
en la **asociación**, no en la detección. Refuerza el parking lot de **asociación
global / tracklets**.
