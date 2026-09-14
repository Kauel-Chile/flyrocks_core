# Funcionalidades — catálogo del sistema

> Documento **acumulativo**: todo lo que el sistema hace hoy, no lo que entró en
> una versión. La columna **Desde** dice en qué entrega apareció cada cosa, y
> **Origen** si salió de este trabajo o del resto del equipo.
>
> Última revisión: 2026-09-14 · Lo que entra en cada versión: `V9.md`, `V9.1.md`.

Verificado comparando los archivos de cada entrega, no de memoria: la vista del
v8 (121 KB), la del v9 (166 KB) y la actual (207 KB).

---

## 1. La vista de limpieza (la pantalla final)

La única pantalla donde el trabajo se mide en horas. Todo acá es **nuestro**.

### Trabajar una trayectoria

| Función | Qué hace | Tecla | Desde |
|---|---|---|---|
| **Aprobar** | La cosecha: la trayectoria sale del lienzo y ya no la tocan los filtros ni un lazo | `Enter` | **v9.1** |
| **Descartar** | A las descartadas, con su razón | `Supr` | v8 |
| **Aislar** | Dibuja **solo** esa trayectoria, con todo lo demás apagado | `I` | **v9.1** |
| **Alisar** | Reajusta el trazo a la curva que la física permite: quita el zigzag del centroide y repara los puntos saltados, en dos pasadas distintas | `S` · `Shift+S` | **v9.1** |
| **Recortar** | Lazo sobre el tramo bueno: descarta la original y crea otra con ese tramo, con métricas recalculadas | `R` | v9 |
| **Unir** | Empalma dos trayectorias partidas completando la curva, no en recta; el tramo reconstruido va punteado y marcado | `U` | v8 · curva en v9 |
| **Dibujar** | Traza una trayectoria a mano donde el detector no vio nada | `D` | v8 |
| **Curva editable** | Manejadores Bézier; sube o baja de uno a dos sin deformar la curva | `C` | v8 |
| **Dar vuelta el sentido** | Intercambia inicio y final de un trazo dibujado. El origen se busca desde el inicio hacia atrás, así que al revés no encuentra su tiro | `V` | **v9.1** |
| **Deshacer** | Cada acción registra su propio inverso (no instantáneas: serían ~2 MB por paso) | `Ctrl+Z` | v9 |

### Trabajar en masa

| Función | Qué hace | Desde |
|---|---|---|
| **Lazo de descarte** | Encierra un grupo y lo descarta. Solo se lleva **lo que se está viendo**, y nunca lo aprobado | v8 · respeta lo aprobado desde **v9.1** |
| **Filtros por métrica** | Tortuosidad, escape relativo, R², largo mínimo — derivados del propio análisis, no fijos | v8 |
| **Filtro por clase** | Enciende y apaga peligrosas / proyección / fuera de vista. Es un control de vista, no de calidad: alcanza también a lo aprobado y a lo editado a mano | v8 · alcanza a todo desde **v9.1** |
| **Descartadas a la vista** | Se muestran en transparencia para revisar qué se fue, y se restauran | v8 |
| **Lo corregido a mano no pasa por los filtros** | Una trayectoria unida, dibujada o recortada ya pasó por el juicio de una persona | v9 |
| **La clase se recalcula al editar** | Recortar, unir, dibujar o alisar cambia dónde termina el vuelo, y la clase se rehace con la misma regla del pipeline. Antes se heredaba | **v9.1** |
| **Lo aprobado es intocable en masa** | Ni los filtros, ni un lazo, ni el recorte se llevan una aprobada. Para sacarla hay que seleccionarla y descartarla a mano | **v9.1** |

### Ver

| Función | Qué hace | Desde |
|---|---|---|
| **Color por avance** | Pinta una sola pregunta: ¿ya pasé por esta o no? Lienzo vs aprobada. El modo de trabajo | **v9.1** |
| **Color por clase** | Pinta la conclusión: peligrosa / proyección / fuera de vista. El modo de entregable | v8 |
| **Lupa de trabajo** | Todo · solo el lienzo (lo que falta) · solo las aprobadas · solo las sin empalme al origen. No cambia el estado de nada | **v9.1** |
| **Fondos** | Máscara de cambios · frame pre-tronadura · **video del clip** · ninguno, con opacidad | v8 · video en v9 |
| **Capas** | Malla de tiros, zonas, líneas de escape | v8 · escape arreglado en v9 |
| **Transporte de video** | Play, ±1 y ±10 frames, línea de tiempo, velocidad | v9 |
| **Salto a la 1ª detonación** | Va al frame donde revienta el primer tiro, según el ancla | v9 |
| **Ajustar vista** | Encuadra todo lo visible | v8 |

### Asociar cada roca a su tiro

| Función | Qué hace | Desde |
|---|---|---|
| **Asociación al pozo** | Proyecta la trayectoria hacia atrás y le asigna su pozo, con el empalme punteado | v8 |
| **Cuña de búsqueda** | Nadir y apertura (`k`, `sigma`) calibrables; la calibración viaja con el caso | v8 |
| **Cruce temporal** | Tiempo de detonación del CSV: retardo máximo y tolerancia sobre el **ancla** del clip | v8 |
| **Ancla ajustable** | Slider que corre el origen temporal de la secuencia | v8 |
| **Selección por pozo** | Marca un tiro y aísla lo que salió de él | v8 |
| **Las que quedaron sin origen** | Cuántas son y por qué —pocos puntos, sin tangente, ningún pozo detrás, o el calce temporal las descartó— con una lupa para verlas y corregirlas | **v9.1** |

### Entregables

| Función | Qué produce | Desde |
|---|---|---|
| **1 · JSON de trayectorias** | El entregable de datos | v8 |
| **2 · Fondo (intensidad)** | La imagen base | v8 |
| **3 · Fondo + trayectorias** | El plano de vuelo | v8 |
| **4 · + empalmes al tiro** | Lo mismo, con cada roca ligada a su pozo | v8 |
| **5 · Heatmap de tiros** | Qué pozos proyectaron más | v8 |
| **6 · Histograma de alcances** | Distribución del alcance: recta del tiro de origen al último punto del trazo, con el radio de evacuación marcado | v8 · mide el alcance desde **v9.1** |
| **Descargar los 6** | Todo de una vez | v8 |

Las exportaciones **fuerzan el color por clase** y apagan la lupa: el modo de
trabajo es para trabajar, el entregable sale siempre igual.

---

## 2. Guardar y retomar

| Función | Qué hace | Desde |
|---|---|---|
| **Guardar avance (archivo)** | Descarga el `.json` con todo el trabajo manual | v8 |
| **Guardar avance (en el análisis)** | Además lo guarda dentro del propio job: retomar no depende de que el archivo siga en Descargas | **v9.1** |
| **Reabrir con avance** | Abrir un análisis que tiene avance lo aplica solo: la pantalla vuelve como la dejaste —umbrales, capas, clases apagadas, modo de trabajo, alisado y aprobadas— | **v9.1** |
| **Ver el análisis original** | Vuelve al resultado crudo del pipeline sin borrar el avance | **v9.1** |
| **Cargar** (dentro de la vista) | Aplica un archivo de avance **sobre el análisis abierto** | v8 |
| **Soltar el archivo** (pantalla de entrada) | Abre el archivo **en su propio análisis**, directo a la vista final | **v9.1** |
| **El avance es de su pasada** | Recalcular con otro percentil borra el avance: los `track_id` se reasignan y aplicarlo sería incorrecto en silencio | **v9.1** |

---

## 3. La pantalla de entrada (`/inicio`) — toda **v9.1**

| Función | Qué hace |
|---|---|
| **Lista de análisis** | Todo lo que hay en el equipo: video, fecha, trayectorias, peso en disco y si tiene avance guardado |
| **Abrir donde quedó** | El terminado va a la edición; el que quedó esperando el corte de ruido vuelve a su slider con la máscara ya calculada |
| **Retomar sin reprocesar** | Reabrir no vuelve a correr el pipeline: son minutos de IA que no se pagan dos veces |
| **Borrar de a uno** | Elimina el análisis y sus archivos, avisando si tiene aprobadas guardadas |
| **No ofrecer lo que no sirve** | Los análisis de versiones anteriores, los que fallaron y los que están procesando se muestran, pero no se abren: dicen por qué |
| **Login → entrada** | El login cae en la pantalla de entrada, no en el paso 1 |

---

## 4. El core

| Función | Qué hace | Desde | Origen |
|---|---|---|---|
| **Carpeta por análisis** | `temp_videos/<job_id>/`: cada máscara, frame y clip con URL propia | v9 | Nuestro |
| **Frame de referencia** | Se extrae tres frames antes de la primera detonación, en JPG | v9 | Nuestro |
| **Clip reproducible** | Derivado H.264 en un hilo aparte: sin él el fondo de video queda negro | v9 | Nuestro |
| **Ancla temporal** | Guarda `frame_detonacion`, `frame_inicio_corte` y el ancla que sale de restarlos | v8 | Nuestro |
| **Malla de tiros** | El CSV de secuencia se proyecta a la imagen y queda en el job, con sus tiempos | v8 | Nuestro |
| **Lista de análisis** (`GET /api/jobs`) | Lo que existe en el equipo, con fecha | v9 | Nuestro |
| **Avance en el job** | Columna `avance` + resumen leído con `json_extract` para no traer megas | **v9.1** | Nuestro |
| **`DELETE /api/jobs/{id}`** | Borra fila y carpeta, con la ruta verificada antes del `rmtree` | **v9.1** | Nuestro |
| **`RETENCION_HORAS`** | Purga por antigüedad, **apagada por defecto**: ahí vive el trabajo del cliente | **v9.1** | Nuestro |
| **Estado de cada análisis** | `abrible`, `motivo`, `bytes` y `retomar_en` para que la lista sepa qué ofrecer | **v9.1** | Nuestro |
| **Corte de ruido interactivo** | El pipeline se pausa, el usuario elige el percentil sobre la máscara y `/api/resume` reanuda | **v9.1** | Del equipo |
| **Nodos nuevos** | Vista previa de percentil, intensidad de ambos signos, fuera de vista por velocidad, fusión de paralelas | **v9.1** | Del equipo |
| **Filtro de humo por IA** | Modelo ONNX, con el consumo de RAM acotado | v9 | Del equipo |
| **Logging y consola** | Los `logger.info` del pipeline se ven, y la consola de Windows ya no mata la corrida con un emoji | v9 | Nuestro |

---

## 5. El wizard (React)

| Función | Qué hace | Desde | Origen |
|---|---|---|---|
| **Pasos 1–3** | Video, recorte y calibración (homografía, zonas, parámetros) | v8 | Del equipo |
| **Paso 4 — corte de ruido** | Slider de percentil sobre la máscara, antes de que el pipeline siga | **v9.1** | Del equipo |
| **Paso 5 — vista de limpieza** | La vista nueva embebida, con la anterior a un clic para comparar | v8 | Nuestro |
| **Ir y volver entre las dos vistas** | El interruptor va en los dos sentidos, ninguna se desmonta (no se pierde lo limpiado), y la anterior se abre con las trayectorias que llevas limpiadas | **v9.1** | Nuestro |
| **Retomar el paso 4 por URL** | `?job=` para volver a un análisis que quedó esperando el percentil | **v9.1** | Nuestro |
| **Guía de matriz, cronómetro, máxima proyección** | Ayudas sobre la calibración y el reporte | **v9.1** | Del equipo |

---

## 6. El paquete

| Pieza | Qué hace |
|---|---|
| `entrega/detovision.bat` | Levanta todo en el equipo del cliente y abre el navegador |
| `entrega/armar_paquete.py` | Arma el ZIP de entrega |
| `entrega/docker-compose.yml` | Los tres servicios, con `IA_ACTIVA` y `RETENCION_HORAS` |
| `debug/publicar_vista.py` | Publica la vista del core a `flyRocks_frontend/public/vista.html` (esa copia es **generada**: no se edita a mano) |
| `debug/job_desde_caso.py` | Siembra un caso congelado como análisis real, para demos sin reprocesar |
