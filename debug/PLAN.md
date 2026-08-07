# Plan — Detección de Flyrocks (enfoque human-in-the-loop)

> Estado: **planificación**. Documento vivo. Última revisión: 2026-07-22.
> Entregable #1 = **máscara visual limpia de trayectorias**. Datos por roca
> (distancias, zonas) = después.

---

## 1. Principio estratégico

No perseguir automatización 100%. Es una tarea **crítica de seguridad, de bajo
volumen y alto valor** (una tronadura a la vez, con tiempo para revisión de un
experto). El camino robusto es **semi-automático**: el usuario entrega *priors*
que el algoritmo respeta como restricciones duras. Diseñamos **"al revés"**:
partimos de lo que el usuario puede aportar, y alrededor construimos los filtros.

---

## 2. Estado actual del pipeline (Camino 2, funcionando)

1. Estabilización de la deriva del dron (optical flow + afín).
2. Extracción de eventos por diferencia de intensidad entre frames (SIN filtro
   de energía — botaba rocas tenues).
3. Exclusión del área de polvo por **mapa de persistencia**.
4. Tracking de velocidad constante con **continuidad física** (sin teletransportes).
5. Filtro temporal (descartar trazas nacidas antes del inicio de tronadura).
6. Filtro de secuencia (traza-back a un tiro ya detonado) — alineación aprox.

**Descartado:** Camino 1 (extracción polar desde un punto) — "falso" con
parábolas y origen multi-punto. El sustrato óptico (diff de intensidad) se
mantiene: la señal existe, las ganancias están en priors + asociación + edición.

---

## 3. Catálogo de inputs del usuario

| # | Input | Qué desbloquea | Dónde | Fase |
|---|-------|----------------|-------|------|
| 1 | Frame exacto de inicio de detonación | Ancla temporal precisa | offline→UI | 0 |
| 2 | Colocar/rotar malla de tiros (= homografía) | Alinear secuencia + distancias en metros | UI | 1 |
| 3 | Fin de detonación | **Derivado del CSV** (no requiere input) | — | — |
| 4 | Pintar áreas: verde=humo, rojo=zona flyrock | Prior espacial + recuperar rocas que salen de DENTRO del humo | offline→UI | 0/1 |
| 5 | Máscaras gris multi-umbral | Recuperar rocas tenues (bug de recall) | offline | 0 |
| 6 | Editar segmentos (unir/borrar/crear) | Limpieza final manual | UI | 2 |
| 7 | Unir traza al tiro más cercano | Coherencia visual del dibujo | algoritmo | 3 |

---

### 3b. Semántica de anotación (detalle del input #4)

Los colores etiquetan **CLASES (intención), no identidades**. Por eso **un color
por clase basta y los cruces no importan** (si 50 trayectorias importantes se
cruzan, todas son azules; separarlas es trabajo del algoritmo, no del usuario).
Nunca se necesitan "infinitos colores".

- 🟢 **Verde = ignorar** (humo / no-interés).
- 🔴 **Rojo = zona con flyrocks** (extraer; **anula la exclusión** de polvo ahí,
  porque las rocas salen de DENTRO del humo).
- 🔵 **Azul = crítico / no perder.** Rol algorítmico: **relajar los filtros**
  (rectitud, largo, energía) para **garantizar la captura** de las importantes
  (las de largo alcance y las **curvas**, que los filtros automáticos tienden a
  descartar). Sirve además de **checklist de validación** de la salida. Se pinta
  como área o como trazo grueso siguiendo una curva particular (semilla).
- (Futuro) 4º color = equipos / zona a proteger, para la etapa de métricas.

**REGLA DEL AZUL (definida por el cliente):** un trazo azul = **UNA sola
trayectoria** (no pueden haber 2+ siguiendo la misma línea azul). Implicancias:
- Todos los fragmentos cuya **mayor parte** esté dentro del trazo pertenecen a esa
  roca y se **fusionan en una** (reconstruyendo su línea central con una curva
  suave). Ojo: NO basta con "tocar" el azul — una recta que solo lo cruza NO
  pertenece (ese bug eliminaba la curva marcada y consumía trazas de más).
- **MECANISMO DE RESCATE:** si al revisar falta una trayectoria, el usuario la
  pinta de azul y el algoritmo la reconstruye desde el set SIN dedup. Ciclo:
  ver -> pintar lo que falta -> recuperar. Implementado en `exp_dedup_v3.py`.

Lienzo para pintar = la **máscara gris de estelas** (se ven las estelas, se pinta
con precisión). **Entrada** = pocos colores (clases, los pinta el usuario);
**salida** = cada trayectoria con color propio, asignado **automáticamente** por
el algoritmo (ahí sí "infinitos colores", pero los pone la máquina).

## 4. Hechos técnicos establecidos

- **CSV `Secuencia (2).csv`**: 129 tiros, `DetonatingTime` en **milisegundos**
  (3000–5769 ms). Toda la tronadura dura **~2769 ms ≈ 83 frames**.
- **Fin de detonación se DERIVA**, no se estima a ojo:
  `frame_tiro = frame_inicio + (tiempo_ms − 3000)/1000 × fps`.
- **La homografía es doble propósito**: colocar la malla = definir H; esa misma H
  da las **distancias en metros** para el reporte de zonas peligrosas.
- Detonación observada: parte ~13.5 s (frame ~30 del clip 12.5–18.5 s), propaga
  **derecha→izquierda** (y abajo→arriba, aprox). Burst pleno ~frame 99.
- **Las rocas emergen desde DENTRO del humo**, no solo del borde → la exclusión
  por persistencia hoy las bota. El área roja pintada debe **anular la exclusión**.

---

## 5. Roadmap por fases

### Fase 0 — Offline, sin UI (validar en ESTE pozo)
Objetivo: probar que los priors del usuario limpian el resultado, sin costo de UI.

- **Frame de inicio:** genero un *contact sheet* de frames numerados (~13–16 s);
  el usuario indica el frame exacto.
- **Homografía:** genero (a) un frame a **resolución completa** y (b) un
  **esquema de la malla con números de tiro** (desde el CSV). El usuario entrega
  **≥4 correspondencias** tiro↔píxel (en Paint) → se calcula H precisa.
- **PNG pintado (verde/rojo):** el usuario pinta áreas sobre la máscara gris.
  Rojo = anular exclusión + priorizar; verde = descartar. Puede hacerlo en
  cualquier editor y pasar el PNG.
- **Bug de recall (punto 5):** investigar por qué trazas claras (visibles en
  percentil 95) no se marcan; probar **detección multi-umbral** y fusión.

### Fase 1 — UI web (reutilizando `index.html` existente)
Solo si Fase 0 demuestra valor. El front actual ya tiene andamiaje (h_matrix,
zonas). Agregar: marcar frame de inicio, **colocar/rotar la malla** (define H),
**pincel** para pintar áreas humo/flyrock. Reutilizable en toda tronadura futura.

### Fase 2 — Edición final de trayectorias
Unir/borrar/crear segmentos manualmente (stitching asistido). Paso casi final.

### Fase 3 — Salida y métricas
- Unir cada trayectoria al **tiro más cercano** (dibujo coherente completo).
- Vía H: **distancias en metros**, clasificación por zona (proyección /
  proyección peligrosa), conteo por sector. Alimenta el reporte PDF.

---

## 6. Decisión web vs offline

**Offline primero (Fase 0), web después (Fase 1).** Los inputs de este pozo son
de una sola vez y baratos (texto + un PNG). No invertir en UI hasta validar que
los priors sirven. Cuando se confirme, la UI se paga sola (se reutiliza siempre).

---

## 7. Parking lot (futuro / opcional)
- **Asociación GLOBAL / tracklets (POR PROBAR — alta prioridad).** Reemplazar el
  tracking greedy frame-a-frame por un enlace global offline (min-cost flow /
  MHT / grafos sobre todas las detecciones a la vez). Ataca de raíz la
  fragmentación/duplicación. Es probablemente lo que hacía la solución del
  cliente (procesamiento digital, ~6 h): cambió velocidad por calidad, con
  asociación global + multi-umbral + ajuste físico. Como esto es offline y de
  bajo volumen, 6 h es aceptable si el resultado es confiable.
- Flujo óptico **denso** para separar humo/roca por campo de velocidad (pesado,
  beneficio incierto — no priorizar).
- Clasificador ML por-trayectoria si se juntan datos etiquetados.
- Anotación de "verdad de terreno" (pocas rocas marcadas) para tuning objetivo.

## 8bis. ETAPA ACTIVA (desde 2026-07-26): refinar la ENTRADA
Se pausa el hilo de reconstrucción de trayectorias azules y se vuelve aguas
arriba, al preprocesamiento: **modelo de fondo del pre-roll** (umbral adaptativo
por píxel + diff contra fondo), **realce de linealidad** (cambiar el eje de
intensidad a forma: línea vs mancha) y **binarización con histéresis** de dos
umbrales, con herramienta de slider. Solo máscaras e imágenes; no se toca el
tracker. → **`debug/PREPROCESO.md`** (decisiones, riesgos y orden de trabajo).

## 8. Camino de las MÁSCARAS (en curso)
Combinar capas complementarias en un mismo análisis: (1) intensidad (qué tan
fuerte), (2) temporal/primera-llegada (cuándo), (3) máscara pintada (criterio
experto), (4) malla de detonación (dónde están los tiros). Cada una tiene huecos
que las otras cubren. Salidas ordenadas en `debug/out/6_mascaras/`.

---

## 8. Qué necesito del usuario para arrancar Fase 0
1. Frame exacto de inicio (tras el contact sheet que yo genere).
2. ≥4 correspondencias tiro↔píxel (tras las 2 imágenes de referencia).
3. Un PNG pintado (verde=humo, rojo=flyrock), aunque sea burdo.
