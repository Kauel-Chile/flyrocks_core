# Pendientes — Flyrocks

> **Índice de temas abiertos.** Este documento es el *menú*: una línea por tema
> para poder elegir cuál retomar. El detalle vive en los docs enlazados.
> Última revisión: 2026-09-14.
>
> Convención: cada tema tiene **estado**, **por qué importa** y **dónde está el
> detalle**. Al cerrar uno, moverlo a "Cerrados" con la fecha.

---

## Activos (lo que está en la mesa ahora)

### P21 — La tronadura en el tiempo: animar y exportar el vuelo ⬅ **pedido 2026-09-04, planificado, sin empezar**
**Estado:** decisiones tomadas con el usuario, nada codificado. *«Esto le encanta
al cliente.»*

Reproducir la tronadura: las trayectorias **finales —las que quedaron después de
limpiar—** dibujándose en el tiempo sobre el canvas, con un botón de play, y
después poder exportarlo como video.

**Los datos ya están, medidos sobre `casos/ia-v9`:**
- El **100%** de las trayectorias trae `frames` punto por punto. No hay que
  interpolar nada.
- El vuelo completo son **453 frames = 15,1 s** a 29,97 fps.
- La secuencia de tiros dura **2,8 s** (113 pozos, de 3.000 a 5.769 ms).
- El ritmo se ve bien solo: nacen 83 rocas en el segundo 1, **469 en el 4** (el
  pico), y decae a 139 en el 12.
- La vista **ya tiene** reproductor de clip, transporte, línea de tiempo,
  selector de velocidad y `irAFrame()`. Falta que los TRAZOS respondan al frame:
  hoy `dibujarTrazos()` pinta la polilínea entera siempre.
- **ffmpeg ya viaja en la imagen del core** (`Dockerfile`), así que un MP4 H.264
  de verdad es posible — no un WebM que después no abre en PowerPoint.

**Decisiones del usuario (2026-09-04):**
- **Sin punta destacada.** El trazo se dibuja avanzando en su color de siempre.
  Una punta brillante *«opacaría la roca real si es que es visible atrás»*: el
  fondo puede ser el clip, y ahí la roca de verdad se está viendo.
- **La detonación del pozo sí se anima** — un brillo cuando revienta.
- **Play en el mismo canvas**, y exportar después. No una pantalla aparte.
- **El video se arma con lo VISIBLE**: las aprobadas más lo que quede en
  pantalla. Misma regla que el resto de la herramienta —*me llevo lo que veo*—.
- **El fondo sigue siendo elegible**, tal como está hoy.
- **Las rocas salen de su pozo**, con el mismo tratamiento que ya usa el canvas:
  **punteado el tramo reconstruido, sólido desde donde se vio de verdad**.

**El calce temporal — esto es lo que hay que resolver bien:**

Una roca tiene DOS tiempos y hoy solo se usa uno.

1. `frame_deton` = `ancla + (t_pozo − t_min) × fps / 1000` — cuándo revienta su
   tiro. Sale del CSV de secuencia, que da tiempos **relativos**; el `ancla` es
   el origen (frame del clip donde detona el primero) y hoy ya se guarda por job
   y se afina con el slider `fAncla`.
2. `frame_nace` = `T.t_ini` — cuándo la **cámara** la vio por primera vez. No es
   cuándo salió: el destello y el humo tapan el arranque, con una **mediana de
   50 frames (1,7 s) de retardo** ya medida.

La animación tiene que usar los dos: **entre `frame_deton` y `frame_nace` se
dibuja el empalme punteado avanzando** (el tramo reconstruido, que es
exactamente lo que ese punteado significa hoy en el canvas), y **desde
`frame_nace` el trazo real, sólido**. Así la roca sale cuando revienta su tiro y
"aparece" cuando la cámara la agarra, que es la verdad de la medición.

Casos a resolver antes de codificar:
- **Trazas que nacen antes que su pozo.** 24 de 3.742 (1%) nacen antes del frame
  del ancla (la primera está en el 14, el primer tiro en el 48). Si
  `frame_nace < frame_deton` no hay tramo que animar hacia adelante. ¿Se dibujan
  sin empalme, se ocultan hasta el ancla, o se corrige el ancla? **Se ve en
  pantalla si no se decide**: rocas volando antes de que reviente nada.
- **Trazas sin pozo asociado**: no tienen `frame_deton`. Se dibujan desde su
  nacimiento y sin empalme, que es lo que ya hacen hoy en el canvas.
- **Con la asociación apagada** (`A.activo = false`) no hay pozos: la animación
  degrada a "cada traza desde que nace", sin empalmes ni brillo de detonación.
- **El ancla se puede validar con la propia animación**: si las rocas salen
  antes o después del fogonazo del clip, el slider está corrido. Es la primera
  vez que ese parámetro tiene una comprobación visual directa.

**El calce de VELOCIDADES es el que valida todo (idea del usuario, 2026-09-04):**

*«Entre que detona el tiro y se hace la proyección hacia donde empieza lo
visible, con su tiempo, las velocidades deberían calzar. Y eso es fácil de ver a
ojo humano: se verán desincronizaciones en caso de haber error.»*

Es el punto más importante del diseño. El tramo punteado NO puede dibujarse a
una velocidad arbitraria —repartir el empalme uniformemente entre `frame_deton`
y `frame_nace` sería exactamente eso—: tiene que ir a la velocidad que la roca
traía. Si el punteado avanza lento y de golpe la traza real sale disparada
(o al revés), eso **es** el error, y se ve sin medir nada. Las fuentes posibles:
el `ancla` corrido, una asociación al pozo equivocada, o la escala px/m.

Y lo que se ve a ojo también se puede **calcular**: con la velocidad inicial `v`
de la traza visible y el largo `L` del empalme, el tiempo implícito de ese tramo
es `L / v` frames. Comparado con `frame_nace − frame_deton` da un residuo por
trayectoria. Sobre todas ellas eso permite (a) **ajustar el `ancla`
automáticamente** minimizando el residuo mediano, en vez de moverlo a ojo, y
(b) marcar las asociaciones cuyo tiempo implícito no calza con su pozo. Vale por
sí solo, aunque el video nunca se exporte.

**Dos números medidos que hay que mirar ANTES de codificar esto** (caso `ia-v9`,
escala 8,55 px/m):

1. **La velocidad inicial está inflada por el ruido.** Medida sobre los primeros
   4 puntos da una mediana de **16,1 m/s**; medida como cuerda/duración de la
   traza completa, **8,4 m/s**. Un factor **1,9×**. Es el mismo zigzag del
   centroide que quita el alisado, y en tramos cortos infla el desplazamiento.
   Si el punteado se dibuja con la velocidad de los primeros puntos, va a ir
   casi al doble de rápido de lo que corresponde. **Conviene medir la velocidad
   sobre la traza alisada, o sobre un tramo largo, no sobre los primeros
   puntos.**
2. **El tramo reconstruido puede ser más largo que el medido.** A la velocidad
   mediana, en el retardo típico de 50 frames la roca recorre **14 m** (43,9 m
   las rápidas), mientras que **el largo mediano del trazo visible es de 5 m**
   (p90: 36 m). O sea: en la mitad de los casos, la mayor parte de lo que se
   verá moverse en el video es **punteado reconstruido, no medición**. Eso no
   invalida la animación —el punteado ya significa "esto no se vio"— pero si va
   a una reunión con el cliente hay que decirlo, y quizá **animar el empalme
   solo para las trayectorias largas**, donde el tramo medido domina.

**Fases:**
1. **Trazos en el tiempo** — el bucle de `dibujarTrazos()` respeta el frame
   actual, con la estela acumulada (la mediana de una traza son **21 frames =
   0,7 s**: sin acumular, el video queda vacío y parpadeante, y además el último
   cuadro deja de ser la imagen final que el cliente ya conoce). Botón de play
   en el canvas. Es el 80% del valor: ya se puede mostrar en pantalla.
2. **Pozos detonando** en secuencia con su brillo, sobre esos mismos 2,8 s, y
   los empalmes punteados saliendo de cada uno **a la velocidad medida de cada
   roca**, no a una velocidad de relleno. Acá es donde se ve si el `ancla` está
   bien: es la primera comprobación visual directa que tiene ese parámetro.
3. **Exportar**. Dos caminos, no excluyentes: `MediaRecorder` sobre el canvas
   (WebM, casi gratis una vez hecha la fase 1) o **el core con ffmpeg** (MP4
   H.264 a 4K, que es lo que sirve de entregable y puede reusar el clip del job
   y el avance guardado, o sea sale de lo aprobado y no del crudo).

### P23 — El histograma medía otra cosa ✅ **hecho 2026-09-14**
**Reportado por un usuario:** editó un caso, quedó con ~300 de 5.000
trayectorias, y el histograma las puso **todas en la primera barra**, como si
ninguna hubiera llegado a 5 m — habiendo trayectorias largas y cortas.

**Qué pasaba.** El botón decía «Histograma de alcances» pero el gráfico no
medía alcance: graficaba `escape_relativo × diametro_equivalente`, o sea cuánto
se salió la roca del **borde del polígono de voladura**. Esa medida vale 0 para
toda roca que no cruzó el borde, y depende de un diámetro que la vista
**inventaba** cuando la zona de origen no llegaba en el job: había un
`Math.max(areaM2, 1)` que lo clavaba en **1.13 m**, y con eso la voladura entera
medía menos de 4 m y el caso completo caía en la primera barra. Sin un error,
sin un aviso. Verificado sobre `casos/3160-789`: sano reparte 726 trayectorias
en 21 barras hasta 206 m; con la zona ausente, 726/726 en la primera y un máximo
de 3.86 m.

**Qué se hizo:**
- El histograma grafica el **alcance desde el tiro de origen** — recta del pozo
  al último punto del trazo, la misma cifra que el entregable llama
  `distancia_m` y resume en `alcance_max_m`. El gráfico y el JSON por fin dicen
  lo mismo. La salida del área sigue en el archivo como `salida_area_m`.
- La fórmula del alcance vive en **una** función (`alcanceDesde`); estaba
  copiada en el panel y en el entregable.
- El diámetro ya no se inventa: sin zona de origen queda `null`, `salida_area_m`
  sale `null` en vez de un cero que se lee como «no salió del área», y queda un
  `console.warn` diciendo qué falta.
- Si hay trayectorias sin tiro de origen, **se avisa antes de exportar**: para
  ésas el gráfico cae a la cuerda del trazo, que mide solo lo que la cámara
  alcanzó a ver (mediana de 5 m) y vuelve a amontonar la primera barra.
- La línea del radio de evacuación ya no revienta la exportación cuando el caso
  no trae radio.

### P24 — Cuáles quedaron sin empalme al origen ✅ **hecho 2026-09-14**
Pedido junto con P23: *«poder decir cuántas están empalmadas y cuántas no, y/o
cuáles, para corregirlas.»*

- Cada trayectoria guarda **por qué** quedó sin origen (`sin_empalme`): pocos
  puntos, sin tangente, ningún pozo detrás del inicio, o el calce temporal los
  descartó. Antes el panel las contaba juntas y les atribuía a todas el mismo
  motivo —el número era correcto y la explicación no—.
- El panel de asociación las desglosa por motivo y trae un botón **«Ver solo las
  sin empalme»**, que es una lupa nueva junto a lienzo/aprobadas.
- El entregable lleva `resumen.sin_empalme` y `sin_empalme_por_motivo`.
- De paso: el archivo asociaba trayectorias que la pantalla se negaba a asociar
  (bajo `nMin` puntos la tangente no es confiable), así que el panel decía 176
  sin empalme y el JSON 75 del mismo caso. Ahora los dos usan la misma regla.

### P25 — El sentido del trazo dibujado y el escape de lo editado ✅ **hecho 2026-09-14**
Salió de revisar el punto que quedaba abierto de P23: las dibujadas a mano
nacían con `escape_relativo: 0` fijo.

**Dar vuelta el sentido.** El sentido de un trazo importa: la asociación al pozo
sale de la tangente **inicial** proyectada hacia atrás, así que una trayectoria
dibujada al revés busca su origen en la dirección contraria y se queda sin
pozo. Medido sobre `casos/3160-789`: la misma curva se asocia al tiro 109 en el
sentido del vuelo y sale «sin pozo detrás» invertida. No había forma de
arreglarlo salvo borrarla y volver a dibujarla.

- Botón **«Dar vuelta el sentido»** en el panel y tecla `V`, con deshacer.
- Solo para trazos **sin eje temporal** (los dibujados). Dar vuelta uno con
  `frames` dejaría la lista en orden descendente, y de ahí salen `t_ini`, el
  calce temporal y el orden de las uniones. Los del detector vienen bien por
  construcción: el tracker sigue al video.
- `invertir()` ya existía pero solo daba vuelta `puntos`, y en una dibujada la
  **Bézier manda** (`tangenteInicial` la lee a ella): invertía la lista sin
  cambiar nada de lo que se ve ni de lo que se asocia. Ahora da vuelta también
  los puntos de control y los índices del empalme.

**El escape relativo se recalcula.** `escapeRelativoDe()` hace en la vista lo
mismo que el nodo `OriginAreaExpansion` del pipeline: distancia máxima a la que
un punto se sale del hull del polígono de origen, dividida por el diámetro
equivalente, todo en píxeles. Validado contra los valores que el pipeline dejó
guardados: **691 de 726 idénticas y las 35 restantes difieren en 0.01**, que es
el redondeo a dos decimales del nodo. Se aplica al dibujar y al mover un tirador
(`rasterizar`), al recortar y al unir —donde además el escape incluye ahora el
tramo reconstruido—. Antes una dibujada salía en el entregable con
`salida_area_m: 0`, que se lee como «esta roca no salió del área».

**Una cuarta de la misma familia:** `distanciaDesdePozo` (el renglón «desde el
tiro X» del panel) era el único que no aplicaba la regla de `nMin`, así que
mostraba una distancia justo encima de su propio aviso «Sin asociar: 3 puntos»
— 101 trayectorias del caso de prueba. Ya usa la misma regla que el resto.

**Nota que quedó verificada de paso:** las trayectorias manuales **sí** se
asocian a un pozo, y **sin** el filtro temporal — no porque haya una regla
especial, sino porque no tienen `t_ini` y la condición `T.t_ini != null` apaga
sola el calce causal. Era exactamente lo que había que hacer; está bien como
está.

### P26 — La pantalla de entrada estaba pintada para fondo claro ✅ **hecho 2026-09-14**
**Reportado:** el texto del pie se ve negro sobre el azul de la página.

`Inicio.page.tsx` usaba los colores del tema **claro** de MUI —`text.secondary`,
`text.disabled`, bordes `#d8d8d8`, hover `#faf6f4`— sobre el `#001223` que
`index.css` le pone a toda la app. Y el proyecto **no tiene `ThemeProvider`**,
así que MUI cae a su tema claro por defecto: hay que decirle el color a cada
superficie, componente por componente. Medido:

| | antes | ahora |
|---|---|---|
| subtítulo y pie | **1.07 : 1** | 7.6 : 1 |
| nota del final | **1.05 : 1** | 5.6 : 1 |
| motivo de un análisis no abrible | 4.5 : 1 (ámbar de fondo claro) | 9.6 : 1 |
| chip «N aprobadas» | 3.1 : 1 | 9.3 : 1 |

Los colores salen del resto del front, no de un gusto nuevo: `#234567` es la
línea que usan Step3, Step4, Step5 y MatrixGuide; el naranja es el de los
sliders del wizard; `#5fafff`, `#ffb726` y `#d80000` son los estados de Step1; y
el cian del chip es el mismo `#22d3ee` con que la vista pinta lo aprobado. El
panel (`#0a1f33`) es el fondo de la app levantado hacia esa línea, para que las
tarjetas se despeguen sin meter un gris ajeno.

De paso: dos reglas horizontales como las que separan secciones en los pasos del
wizard, y el naranja de los textos aclarado a `#ff7a45` — `#f94600` sobre el
azul queda en 4.3:1, justo en el límite.

Verificado con `tsc -b --force`, `vite build` y un render fuera del navegador
(`react-dom/server`) que confirma que los colores que llegan al HTML son los
nuevos.

### P27 — Un lazo se llevaba lo aprobado ✅ **hecho 2026-09-14**
**Reportado:** una trayectoria aprobada se puede eliminar con el lazo o con el
recorte, y si se elimina el contador de aprobadas no se mueve.

Las dos cosas eran ciertas, y la segunda escondía a la primera. Medido sobre
`casos/3160-789` con 20 aprobadas y un lazo que cubre toda la pantalla:

| | antes | ahora |
|---|---|---|
| aprobadas que sobreviven al lazo | **0 de 20** | 20 de 20 |
| contador tras el lazo | 0 | 20 |
| entregable tras el lazo | `activas: 0, aprobadas: 20` | `activas: 20, aprobadas: 20` |

**Lo aprobado no se lo lleva una herramienta de masa.** Era la misma garantía
que los filtros ya daban —«una aprobada no puede escaparse de la pantalla por
mover un slider»— y que a `atrapadas()` y a `aplicarRecorte()` les faltaba.
Aprobar significa que una persona miró esa trayectoria entera y se la lleva; un
lazo no puede deshacer eso de refilón. En el recorte pesa todavía más: descarta
la original y crea otra con id nuevo (`Xr`), o sea que no le quita la aprobación
— la reemplaza por una trayectoria distinta, que ya no es la que se revisó.

- Restaurando **sí** se tocan: devolver a la vida una aprobada descartada no le
  quita nada a nadie.
- Para sacar una aprobada hay que seleccionarla y descartarla a mano, que es
  explícito y reversible. Eso sigue funcionando.
- El lazo y el recorte **lo dicen** cuando respetaron alguna: callarlo cuando el
  lazo agarra tres y deja dos se lee como que la herramienta falla.

**El contador.** `aprobadas` contaba `T.aprobada` sin mirar el estado, así que
una aprobada descartada seguía sumando y el número no bajaba nunca. El caso
extremo lo dejaba absurdo: un entregable con `activas: 0` y `aprobadas: 20`. Una
sola definición ahora, `esAprobada = aprobada && estado !== "descartada"`, usada
por el contador, la cosecha y el resumen del entregable. La marca **se
conserva** en la descartada —descartar es reversible y al restaurarla vuelve
aprobada sin tener que volver a mirarla—, solo deja de contar como cosecha.

El core tenía el mismo error al resumir el avance guardado
(`main.py`, `PUT /api/jobs/{id}/avance`), así que el chip «N aprobadas» de la
pantalla de entrada arrastraba la misma cifra inflada. Corregido con la misma
definición.

### P28 — El interruptor entre las dos vistas iba en un solo sentido ✅ **hecho 2026-09-14**
**Reportado:** desde la vista del compañero no hay botón para volver a la nuestra.

`Step4Bifurcacion.tsx` dibujaba la barra con el botón **solo en la rama de la
vista nueva**. Al pasarse a la anterior devolvía `<Step5 />` pelado: sin barra,
sin botón, y el único camino de vuelta era recargar la página — que además se
llevaba la limpieza no guardada.

Buscando eso aparecieron dos más en el mismo botón:

1. **Cambiar de vista mataba el iframe.** El componente devolvía otra cosa, así
   que React desmontaba la vista de limpieza entera y con ella todo lo trabajado
   desde el último «Guardar avance». Ahora **las dos quedan montadas** y solo se
   alterna cuál se ve; la anterior se monta la primera vez que se pide y desde
   ahí se queda (montarla de entrada cuesta un canvas y un OpenCV que casi nadie
   usa, y desmontarla al volver le borraría sus ediciones).
2. **Peor: te expulsaba del asistente.** `Step5` arranca con
   `if (!step1Data) navigate("/wizard/step-1")`. Quien abrió el análisis desde la
   pantalla de entrada no tiene esos datos, así que el botón lo mandaba al paso 1
   y perdía el trabajo. Ahora el cambio se ofrece deshabilitado y con el motivo,
   en vez de llevarlo a un callejón.

Además: la barra dice **en cuál de las dos estás** (de lejos se parecen), y al
entrar a la anterior se avisa que **no guarda nada** — lo que se edite ahí no
queda ni en el análisis ni en un archivo.

De paso, la paleta de las pantallas nuestras se movió a `src/ui/paleta.ts`: la
barra y la pantalla de entrada tienen que verse iguales, y con los colores
escritos a mano en cada archivo ya habían empezado a separarse.

### P30 — El filtro por clase y la clase de lo editado ✅ **hecho 2026-09-14**
**Reportado:** ocultar por clase funciona en un video nuevo, pero al cargar un
trabajo anterior «como que no funciona». Y: al editar una trayectoria habría que
recalcular a qué clase pertenece.

**Las dos cosas eran ciertas y tenían la misma raíz:** la clase se trataba como
si fuera un dato fijo del pipeline.

**1 · El filtro por clase estaba entre los umbrales de calidad.** `recalcular()`
exenta de los filtros lo aprobado y lo tocado a mano —bien: son juicios sobre la
calidad de una detección automática, y una persona ya decidió— pero la casilla
de clase estaba dentro de esa exención. Al abrir un trabajo guardado, **todo lo
aprobado ignoraba el filtro**, que es justo lo que uno tiene cargado. Medido
sobre `casos/3160-789` con 200 aprobadas:

| apagar «Proyección» | antes | ahora |
|---|---|---|
| desde cero | 0 de esa clase visibles | 0 |
| sobre trabajo cargado | **141 visibles, las 141 aprobadas** | 0 |

El filtro por clase **no es un juicio de calidad**: no dice «esta traza es mala»,
dice «enséñame solo las peligrosas». Es un control de vista, y ahora se aplica a
todo. Esconder una aprobada por clase no le quita nada: sigue aprobada, sigue
contando y vuelve al marcar la casilla.

**2 · La clase ahora se recalcula al editar.** `clasificarDe()` aplica en la
vista la misma regla que el nodo `TrajectoryCategorizer`: «Fuera de vista» si el
último punto quedó pegado al borde del cuadro (5 px) o si prolongando su
velocidad terminal 30 frames se saldría; si no, «Proyección» dentro de la zona
de seguridad y «Proyección peligrosa» fuera. Se rehace al dibujar y al mover un
tirador, al recortar, al unir y al alisar — junto con el escape relativo, en una
sola función (`reMedirGeometria`).

Antes: una unida heredaba la clase del tramo que terminaba último, una recortada
la de la original **aunque el recorte le quitara justo la punta por la que era
«Fuera de vista»**, y una dibujada nacía «Proyección» fija cayera donde cayera.
La clase manda en el color, en el filtro, en la censura del histograma y en el
entregable. Comprobado: recortando al primer 25% las 4 «Fuera de vista» largas
del caso, quedan 3 «Proyección» y 1 «Proyección peligrosa» — antes las 4 seguían
diciendo «Fuera de vista».

**Dos cosas que salieron al validar:**

- El chequeo predictivo **solo se aplica con eje temporal**. Mide velocidad, y un
  trazo dibujado no tiene: se rasteriza a 48 puntos fijos, así que el «paso por
  punto» no dice nada de la rapidez. Suponer 1 frame por punto hacía que casi
  toda curva dibujada saliera «Fuera de vista». Para esas queda el chequeo
  estático, que es geometría pura.
- **El caso congelado `3160-789` es anterior al chequeo predictivo** que agregó
  el equipo (commit `8f35d48`). Con la regla estática sola, `clasificarDe`
  reproduce **726 de 726**; con la regla actual completa coincide en 644 (88,7%)
  y las 82 diferencias son todas hacia «Fuera de vista», que es exactamente lo
  que ese chequeo agrega. O sea: la implementación es fiel, el caso es viejo.
  En un análisis viejo, una trayectoria recién editada queda clasificada con la
  regla nueva y sus vecinas con la vieja. Es el precio de no inventar una
  herencia falsa, y solo afecta a lo que se edita.

**Queda anotado, sin tocar:** un lazo de descarte alcanza también a las
trayectorias que un filtro tiene **ocultas** (`atrapadas` solo excluye las
descartadas), mientras que el recorte sí las respeta (`estado !== "activa"`). Es
anterior a todo esto y nadie lo ha reportado, pero las dos herramientas deberían
decidir igual.

### P22 — Retomar donde quedaste ✅ **hecho 2026-09-11**
> Todo esto viaja en la entrega **v9.1** (así la bautizó el equipo en la
> reunión interna del 2026-09-11: se llamaba v10, pero se prefirió no subir
> tanto de versión). Lo que entra: `entrega/V9.1.md`. El catálogo completo de
> funcionalidades, con desde qué versión existe cada una:
> `entrega/FUNCIONALIDADES.md`.

**Estado:** implementado y probado contra el core corriendo. Salió de traer los
cambios del equipo (25-08 al 08-09) y revisar qué se rompía con lo nuestro.

**Lo que trajo el equipo y cambia el mapa:**
- El pipeline ahora corre en **dos fases**: se pausa en
  `ESPERANDO_PERCENTIL_USUARIO` (18%), el usuario elige el corte de ruido en un
  slider sobre la máscara (`Step4.tsx`, nuevo paso React) y
  `POST /api/resume/{job_id}` reanuda.
- Por eso **el wizard tiene un paso más**: la vista nueva (el iframe) pasó de
  `step-4` a **`step-5`**. Todo lo que apunte al paso 4 abre hoy otra pantalla.
- El blast detector trae `fix thumbnails`; nada nuestro toca eso.

**La regla que ordena esto (decisión del usuario, 2026-09-11):** el wizard se
comporta como un wizard. Volver atrás pierde lo de adelante y *da igual*: los
pasos 1-4 son rápidos. Lo único caro es la **edición manual del paso 5**, que
son horas — «te fuiste a almorzar» no puede costar rehacer el análisis. Y dos
pasadas distintas **no son compatibles**: si cargas a mano el archivo de otra
pasada, verás trayectorias que no calzan, y eso es decisión de quien lo carga.

**Lo que se hizo:**
- `/api/resume` borra el `avance` junto con `json_data`, sin aviso. Los
  `track_id` se reasignan al recalcular: el avance de la pasada anterior apunta
  a rocas que ya no son esas — aplicarlo sería incorrecto **en silencio**, el
  mismo modo de falla de la máscara global.
- `/api/jobs` devuelve **`retomar_en`** (`edicion` | `percentil` | `null`) y un
  `motivo` honesto por caso. La pantalla de entrada **ya no esconde** los
  análisis a medias y abre cada uno donde quedó.
- `Step4` acepta **`?job=`** por URL. Antes dependía de `step3Data.idProjection`,
  que vive solo en memoria: cerrar el navegador ahí dejaba el análisis colgado
  con la fase cara ya corrida. Al retomar abre su propio WebSocket y entra solo
  a `step-5` cuando termina.
- `DELETE /api/jobs/{id}` mira el **status**, no `is_running`: un job pausado la
  tiene en `True` a propósito (mantiene vivo el WebSocket de progreso), y eso lo
  hacía imborrable e invisible para siempre.
- Un análisis que revienta ya no se ofrece: «Terminó con error», y se puede
  borrar.

**Límite conocido:** los pasos 1-2 no tienen job todavía (el video está en el
blast detector), así que retomar vale **del paso 3 en adelante**.

**Pendiente de esta tanda:** el commit `6350edd` borró 147 líneas de comentarios
de `src/main.py` — repuestos en versión corta el 2026-09-11 — y otras 34 en
`services.py`, 8 en `event_extractor.py` y 5 en `ai_smoke_filter.py`, **sin
reponer**.


### P18 — Recortar trayectorias: quedarse con el tramo bueno ⬅ **pedido 2026-08-18, esperando go**
**Estado:** anotado, sin empezar. El usuario da el go mañana.

Hoy una trayectoria es todo o nada: el lazo la descarta entera. Pero hay trazas
que están **bien en un tramo y mal en otro** —un zigzag imposible a la mitad, o
una cola que se fue con el humo— y descartarlas completas tira información
buena.

**Lo que se pide:** poder cortar la parte que no corresponde a la realidad,
**quedarse con el resto y seguir trabajando con eso** (filtros, asociación al
pozo, entregable). Es decir, el recorte tiene que producir una trayectoria de
primera clase, no una anotación cosmética.

Preguntas a resolver antes de codificar:
- ¿La herramienta es el mismo lazo (que corta lo que rodea) u otra distinta
  (tijera: dos clics sobre el trazo y se elimina el tramo entre ellos)?
- Si el corte parte una traza en dos tramos válidos, ¿quedan dos trayectorias
  independientes o una con hueco?
- El `estado`/`razon` del modelo actual es por trayectoria completa. Recortar
  obliga a que sea **por punto**, o a materializar la traza recortada con su
  origen anotado (`fuente: "recorte de <id>"`), como ya se hace con las
  dibujadas a mano.
- El entregable JSON tiene que decir qué se recortó: si el cliente pregunta por
  qué una traza mide la mitad, la respuesta no puede ser «alguien la editó».

### P19 — Unir trayectorias completando la parábola ✅ **primera versión 2026-08-20**
**Estado:** hecho y probado por el usuario. *«Está mejor que antes pero podría
mejorar más; en honor al tiempo déjalo así por ahora.»*

`unirTrayectorias` empalmaba los dos extremos **en línea recta**. Ahora el hueco
se rellena con una curva apoyada en las dos tangentes: la de salida del primer
fragmento y la de entrada del segundo.

Cómo quedó:
- **`tangenteExtremo(T, alFinal)`** — nueva. Ajusta una **cuadrática** por
  mínimos cuadrados sobre todo el tramo (hasta 200 px) y evalúa su derivada en
  la punta. `tangenteInicial` (PCA, recta) se quedó intacta para la asociación:
  ahí interesa de dónde viene la roca en promedio, no la pendiente en el borde.
- **Punto de control = intersección de las dos tangentes.** Una parábola es
  exactamente la Bézier cuadrática cuyo control es donde se cruzan sus
  tangentes, así que no hay constante que calibrar.
- **Techo a la flecha del arco** (L/2). Sin él, con ruido alto las tangentes se
  cruzan lejísimos y sale un rulo peor que la recta.
- Los puntos generados entran en la traza con su frame, se dibujan **punteados**
  y viajan en el entregable bajo `empalme`.

**Lo que las mediciones dijeron, y que sigue abierto.** Sobre parábolas
sintéticas el empalme es exacto (0,00 px contra 8,00 px de la recta). Sobre
**trazas reales** es al revés: partiendo 1.394 trazas y reconstruyendo el hueco,
la curva quedó peor que la recta en el 70% de los casos (mediana 10,45 px contra
7,35 px), y solo empata en el tercio más curvado.

La causa está medida: el ruido de detección es de **2,3–3,1 px RMS sobre tramos
de 21–29 puntos**. Estimar una tangente y extrapolar curvatura sobre eso
amplifica el ruido; la recta no extrapola nada. En la mitad de los casos las dos
tangentes ni siquiera se cruzan hacia adelante.

Se dejó igual porque el pedido era **visual** —que no haya quiebre en V— y en eso
funciona. Pero la fidelidad punto a punto no mejoró, y eso vuelve en P20.

### P29 — Compartir el avance entre las dos vistas ✅ **la ida, hecha 2026-09-14** · la vuelta, descartada
**Pregunta del usuario:** que al pasar de una vista a la otra se conserve el
trabajo, como pasa hoy con el avance del pulido de trayectorias.

**Lo que hay hoy, medido en el código:**

| | vista de limpieza (nuestra) | vista anterior (`Step5.tsx`) |
|---|---|---|
| de dónde lee | `GET /api/results/{job}` **y** `GET /api/jobs/{job}/avance` | `step3Data.projections`, del contexto del wizard |
| qué guarda | archivo `.json` **y** `PUT /api/jobs/{job}/avance` | **nada**: solo descarga CSV y PNG |
| identidad de cada roca | el `track_id` del pipeline | **`Math.round(Math.random() * 100000)`** |
| qué campos conserva | puntos, frames, clasificación, métricas, `estado`, `razón`, `aprobada`, `alisada`, `bezier`, `empalme`, `unida_de`, `recorte_de`, `asociación` | puntos, clasificación, distancia, 3 métricas y un color aleatorio |
| ediciones | en el modelo, y viajan al archivo | en estado de React (`listDelete`, `listConfirm`, `bezierPoints`…), mueren al desmontar |

**El punto que decide.** `wizardDataContext.tsx:525` le asigna a cada trayectoria
un `id_roca` **aleatorio** y tira el `track_id`. Sin identidad estable no hay
forma de decir «esta trayectoria es la que aprobaste»: en cada recarga la misma
roca cambia de id. Y `Step5` no persiste, así que hoy no hay estado suyo que
traer de vuelta aunque quisiéramos.

**Recomendación:**

- **Una vía (nuestra → la del compañero): HECHA.** Al pulsar «Ver con la vista
  anterior», el wizard le pide a la vista de limpieza su estado **actual** por
  `postMessage` (`flyrocks:dame-avance` → `flyrocks:avance-actual`), descarta lo
  descartado y lo mapea al modelo de `Step5`. Se le pide a la vista y **no al
  core** a propósito: el core solo tiene el último «Guardar avance», y lo que
  importa es lo que hay en pantalla ahora.
  - Medido sobre `casos/3160-789`: sin limpiar se llevan las 726; tras descartar
    391 con un lazo, se llevan 335. El mensaje pesa 0,4 MB.
  - Van con el **color por clase** en vez del color aleatorio con que nacen ahí,
    así las dos vistas pintan lo mismo del mismo color.
  - `Step5` les aplica encima sus propios sliders, así que puede mostrar menos
    de las que recibe (98 de 335 con sus valores por defecto). El aviso lo dice,
    o parece que el traspaso perdió trabajo.
  - No entiende `aprobada` ni las Bézier: las pinta como polilíneas.
- **La vuelta: descartada, y en su lugar un aviso.** Exigiría que `Step5`
  conserve el `track_id`, que persista, y que aprenda `estado`, `razón`,
  `aprobada`, `alisada`, `empalme`, `unida_de` y `recorte_de` — o cada paso por
  ella **destruiría trabajo en silencio**, que es peor que no compartir nada. Al
  volver se pide confirmación diciendo exactamente qué se pierde (lo editado
  allá) y qué no (la limpieza, que sigue viva porque el iframe nunca se
  desmonta — ver P28).
- **Lo que de verdad hay que decidir antes es cuál de las dos vistas se queda.**
  El propio comentario de `Step4Bifurcacion` dice que el interruptor existe «en
  vez de decidir hoy cuál gana». Gastar el rehacer del modelo de `Step5` para
  después apagarla sería tirar el trabajo; y si la que gana es la de ellos, el
  puente correcto es el contrario.

### P20 — La unión como parábola editable ⬅ **pedido 2026-08-20**
**Estado:** anotado, sin empezar.

Sigue de P19. Hoy el empalme es una curva **fija**: se calcula al unir y no se
puede tocar. Lo que se pide es que la unión **se transforme en una parábola
editable**, con los mismos tiradores que la herramienta de dibujar trayectoria
(`bezier` + `agarrarNodo` + `alternarGrado`), para deformarla a mano hasta que
tenga sentido físico.

Eso resuelve de raíz lo que las mediciones dejaron abierto en P19: si el ajuste
automático no acierta —y con este ruido no acierta—, que lo corrija una persona.
Encaja con la regla que ya rige: **lo tocado a mano no pasa por los filtros**.

El usuario lo enmarcó más amplio: *«al final todo deberían ser rectas o
trayectorias con forma físicamente posible»*. O sea, no es solo la unión — es
**limpiar y regularizar todas las trazas** para que ninguna tenga una forma que
una roca no pudo volar. La unión editable es la primera pieza.

Va de la mano con P18 (recortar): recortar el medio y reconectar necesita
exactamente la misma curva editable.


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

**~~Bug del core~~ ✅ ARREGLADO 2026-08-13.** Los `print` con emoji reventaban el
pipeline entero con `UnicodeEncodeError` en consola Windows sin UTF-8 (mató una
corrida en el nodo 12, tras 98 s; a ellos no les pasa porque corren en Docker).
Se arregló **en el punto de entrada** (`src/main.py`), reconfigurando `stdout` y
`stderr` a UTF-8 con `errors="replace"`, y no quitando los cinco emojis: los
prints se siguen escribiendo y el próximo emoji volvería a reventar. `debug/
caso_export.py` ya traía el mismo fix, que es la razón por la que exportar casos
nunca falló y sí fallaba el core nativo.

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

**~~La caché crece sin límite~~ ✅ PURGA HECHA 2026-08-13.** Techo configurable
`CACHE_MAX_MB` (2 GB por defecto ≈ 30 configuraciones; 0 = sin límite), y se
purga al **final** de cada corrida — así lo recién calculado ya cuenta y una
purga lenta no retrasa el primer nodo.

Dos cosas que el test dejó claras (`debug/probar_purga.py`, 16 comprobaciones):

- **No se puede borrar por entrada.** Los objetos están deduplicados: varios
  índices apuntan al mismo `objetos/<hash>.bin`, que es justo lo que baja una
  corrida de 659 MB a 60 MB. Es mark & sweep — se descartan entradas por
  antigüedad y después se borra **solo lo que ya no referencia nadie**.
- **La entrada más nueva nunca se purga.** Con un techo menor que una corrida,
  la versión inicial borraba lo que el pipeline acababa de escribir: la caché
  quedaba escribiendo y borrando lo mismo en cada vuelta, sin dar nunca un HIT y
  pagando siempre el costo de escribirla. Ahora se pasa del techo y lo dice en
  el log.

Que la purga falle no puede costar el resultado del pipeline, que es lo caro:
va envuelta en try/except y solo registra un aviso.

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

**El CSV de secuencia ya llega al core ✅ (2026-08-12).** Era el último eslabón:
el navegador lo leía en el paso 3 y nunca salía de ahí, así que un job **no traía
la malla** y sin malla no hay asociación.

- `/api/analyze` acepta `detonation_sequence` (**opcional**: sin él el análisis
  corre igual y un wizard viejo sigue funcionando sin cambios).
- `utils/malla.py` (nuevo) parsea y proyecta con la `h_matrix` que ya viajaba en
  la misma petición. **Verificado: 113 pozos, desvío máximo 0.0 px** contra la
  calibración manual de `pre_tiros.py`.
- En `entrada` van **las dos cosas**: `secuencia.csv` crudo (4 KB, fuente de
  verdad — permite reproyectar si mañana cambia el método sin volver a pedirle
  el archivo al usuario) y `malla.pozos` ya proyectado.
- **Falla blando:** un CSV mal formado anota `entrada.malla_error` y el pipeline
  sigue. Perder la asociación es malo; perder también las trayectorias por una
  fila rara de Excel sería peor.
- En el wizard fue **una línea** (`wizardDataContext.tsx`): el archivo ya estaba
  en el estado y ya se mandaba a `/api/generate_report`; la línea estaba escrita
  y comentada.

**Bug encontrado de paso en la vista:** la homografía llega **aplanada**
(`[[9 números]]`, porque el wizard la manda con un `toString()`) y `escalaDeH`
la leía como 3×3 anidada → `NaN` en silencio → escala 1 px/m. Se normaliza con
`aH3()`, igual que el `reshape(3,3)` que ya hacía el core.

**~~Ancla temporal~~ ✅ CABLEADA 2026-08-13.** El wizard manda ahora los dos
frames del paso 2 (`frame_detonacion` y `frame_inicio_corte`, ambos opcionales),
el core guarda los crudos **y** la resta en `entrada.recorte`, y la vista arranca
el control «1ª detonación» en ese valor en vez de en 0. El control pasa de ser
una adivinanza a una corrección fina.

Se guardan los tres números, no solo el ancla: los crudos permiten recalcularla
si mañana cambia el criterio, sin volver a pedirle nada al usuario — el mismo
principio que guardar el CSV crudo además de la malla proyectada. Si el job no
los trae (un wizard viejo), la vista cae a 0 como antes.

El razonamiento original, que sigue valiendo:


El CSV da tiempos *relativos* entre pozos, no el frame del video donde arranca la
secuencia. Hoy la vista trae un slider («1ª detonación») que el usuario mueve a
ojo. Pero **el sistema ya conoce el número y lo tira**, igual que pasaba con el CSV:

- El blast detector **ya detecta** la detonación (`services/analysis.py:69`), y
  reporta a propósito **6 frames antes** de donde dispara
  (`f_detonacion = (frame_idx - 1) - buffer_frames.maxlen`, con `maxlen = 5`),
  para que el corte no se coma el destello.
- El recorte usa el marcador **del usuario**, no la sugerencia:
  `inicio_frame_usuario` (`routers/video.py:137`). Así que el desfase real depende
  de cuánto lo haya arrastrado.
- Ninguno de los dos viaja al core: mueren en el navegador en el paso 2.

Con esos dos números el ancla sale calculada y el control pasa a ser una
corrección en vez de una adivinanza:

    ancla = f_detonacion_detectado - frame_inicio_del_corte

Si el usuario acepta la sugerencia tal cual, el ancla vale **~6 frames**. El 48
del caso `3160-789` es nuestro: ese clip lo cortamos a mano (12,5 s → 27,6 s) con
1,6 s de aire por delante, no lo produjo el blast detector.

Detectar el destello desde el core ([[P8]]) sigue siendo válido, pero es el
camino caro para un dato que ya está medido aguas arriba.

### El wizard ya no pierde el análisis al recargar ✅ **2026-08-13**

Tres cosas que hacían del paso 3 un callejón sin salida, todas del flujo
original (no las introdujimos nosotros):

- **Un F5 dejaba el análisis inalcanzable.** El estado del wizard son `useState`
  en memoria; al recargar se perdía el `job_id` aunque el core tuviera el job
  entero. Ahora el id queda en `localStorage` al lanzarlo y el paso 4 lo retoma,
  avisando que es trabajo recuperado. Funciona **porque** el core ya guarda su
  contexto completo ([[P17]]): con el id basta para reconstruir la vista.
- **El paso 3 rebotaba al 4.** Su `useEffect` de navegación corre también al
  montar y nadie limpiaba `projections`, así que al volver saltaba de inmediato
  al paso 4: no se podían tocar los parámetros sin reiniciar el wizard. Ahora
  navega solo cuando el análisis **termina**, no cuando el paso se monta.
- **Y el botón «siguiente» estaba muerto al volver.** El disparo era
  `if (csvFile)`, con `csvFile` en estado local del paso, que se pierde al
  desmontarse. Se usa también `fileCsv` del contexto, que sobrevive.

Al relanzar se sueltan las proyecciones anteriores: si no, se mostraban las
viejas como si fueran de la corrida en curso.

**Ojo:** relanzar **crea un job nuevo**, no reusa el anterior — sube el video de
nuevo y corre el pipeline entero. El job viejo queda huérfano hasta que el
limpiador lo barre. Que se pueda reprocesar sin reiniciar el wizard es la mejora;
que sea barato, no.

---

**Observación sobre el front del colega:** `Step3.tsx:190-195` mapea solo
`X, Y, Z, Label` — **descarta `DetonatingTime` al leer**. Por eso en su vista no
se puede hacer calce temporal aunque se quisiera. Con este cambio el tiempo llega
igual al backend por el archivo crudo, así que no es urgente.

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

**El compose canónico ya está decidido ✅ 2026-08-13:** vive en
`flyrocks_core/entrega/docker-compose.yml`, con rutas planas — la forma que
tiene dentro del paquete del cliente, que es idéntica a la de
`detovision_standalone/`.

Se revisó el paquete real (`Detovision_V3/Detovision/`) y **confirma P16 al pie
de la letra**: su compose no tiene volúmenes, ni retención configurable, y el
blast detector sigue bajo su nombre de la v3 (`./flyrocks`). El `.bat` hace
exactamente lo que decía la nota — `docker compose down` al iniciar y al
detener, `up -d --build` para levantar — así que **el arreglo funciona sin
tocarlo**.

`entrega/armar_paquete.py` genera el paquete completo desde el workspace: 13,7 MB
contra los varios GB de acá. Lo que decide, y por qué, en `entrega/README.md`.
Dos cosas que no son obvias y que el script resuelve: **republica la vista antes
de copiar** (viaja como copia generada en `public/`, y si no se enteraría nadie
hasta que el cliente la abre) y **deja viajar el `.env` del frontend**, porque
Vite resuelve las `VITE_URL_*` en tiempo de build dentro del contenedor y sin él
la app se construye sin backend.

**Queda una duplicación:** el compose de la raíz del workspace tiene las mismas
tres correcciones pero con rutas `./detovision_standalone/<servicio>`, para
levantar en desarrollo. Son dos archivos que van a divergir; hay que decidir si
el de desarrollo se deriva de este o se elimina.

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

**~~Deuda del asociador duplicado~~ — NO APLICA, verificado 2026-08-13.** Estaba
anotado que `salidas.py` consumía un entregable generado por un script Node que
replicaba el asociador, y que ambas implementaciones iban a divergir. Al ir a
extraer `demo/asociacion.js` resultó que **ese script no existe en el repo**: fue
temporal y no quedó. Hoy `salidas.py` consume `entregable.json` tal como lo
exporta la vista, así que en el camino del entregable hay **una sola**
implementación del asociador y no hay nada que extraer.

Lo que sí queda es `debug/test_asociacion.py`, que replica el algoritmo — pero de
`demo/trayectorias.html` (la demo vieja de trazos a mano), y para correr el banco
de sintéticos, no para producir entregables. Si algún día se toca la cuña o el
softmax, ese archivo hay que actualizarlo a mano: es el único punto de deriva que
queda, y es de un experimento, no de la salida al cliente.

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
- **`invertir()` muta el tramo original al unir sin eje temporal**
  (`vista.html:1970-1971`). Los tramos unidos quedan *descartados*, no borrados,
  justo para poder deshacer la unión — pero en la rama geométrica el original ya
  quedó dado vuelta, así que restaurarlo no devuelve lo que había. Solo afecta
  trazos dibujados a mano (los del pipeline siempre traen `frames` y nunca se
  invierten). Arreglo: invertir sobre una copia. Anotado 2026-08-13, sin urgencia.
- **El guardado es manual y no hay red de seguridad.** «Guardar avance» descarga
  `caso_<nombre>_editado.json` y «Cargar» lo reabre — pero no hay persistencia
  automática (verificado: ni `localStorage` ni `beforeunload`). Un F5 o cerrar la
  pestaña borra descartes, uniones, trazos manuales, `k`, nadir y ancla **sin
  aviso**. Que recargar devuelva el caso virgen es útil y hay que conservarlo; lo
  que falta es que no se pierda trabajo por un atajo mal apretado. Mínimo:
  `beforeunload` si hay ediciones sin guardar. Anotado 2026-08-13.
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
| 2026-09-11 | P22 — retomar donde quedaste: merge con el equipo, la lista como único camino de vuelta y el avance atado a su pasada |
| 2026-09-14 | P23 — el histograma graficaba la salida del área y no el alcance; el diámetro se inventaba cuando faltaba la zona de origen |
| 2026-09-14 | P24 — motivo de cada trayectoria sin empalme, lupa para verlas y misma regla en pantalla y en el archivo |
| 2026-09-14 | P25 — dar vuelta un trazo dibujado (tecla V) y recálculo del escape relativo en lo editado |
| 2026-09-14 | P26 — la pantalla de entrada usaba el tema claro de MUI sobre el fondo azul: textos ilegibles |
| 2026-09-14 | P27 — el lazo y el recorte se llevaban lo aprobado, y el contador de aprobadas no bajaba nunca |
| 2026-09-14 | P28 — no había vuelta desde la vista del compañero; cambiar de vista mataba el iframe y podía expulsarte del asistente |
| 2026-09-14 | P29 — la vista anterior se abre con lo que llevas limpiado; la vuelta avisa en vez de mezclar modelos incompatibles |
| 2026-09-14 | P30 — el filtro por clase no alcanzaba a lo aprobado; la clase se recalcula al editar |
