# DETO1 — replicar el proceso del cliente

> Estado: **corre de punta a punta, el resultado todavía no son rocas.**
> Iniciado 2026-08-16. Barrido de parámetros lanzado esa madrugada.

## 1. Qué hace el cliente, según nos lo describió

1. Saca una máscara mirando la **diferencia de color entre frame y frame** —igual
   que nosotros—, pero la pinta **binaria** (blanco y negro), no en escala de grises.
2. Sobre esa máscara **pinta las trayectorias que no salieron** originalmente.
3. El proceso completo tarda **~6 horas**, offline.

De (3) dedujimos hace semanas que su asociación es **global** (todas las
detecciones a la vez) y no frame a frame como la nuestra. Eso sigue siendo
inferencia nuestra: nunca lo confirmaron.

## 2. La pregunta que originó esto

> *«En teoría, si bien demora más, no debería perder trayectorias, ¿o no?»*

**No: pierde, y el dato del propio cliente lo demuestra.** Si tenían que pintar
encima las que no salieron, es porque el proceso se las comía.

El binario es una decisión **irreversible por píxel, tomada al principio de
todo**. Una estela tenue no alcanza el umbral y desaparece del universo; ninguna
cantidad de cómputo posterior la recupera, porque el dato ya no existe. Las 6
horas no compran sensibilidad.

Lo que compran es otra cosa, y no hay que confundirlas:

| etapa | qué decide | quién va mejor |
|---|---|---|
| **detección** (umbral) | qué píxeles existen | **nosotros** — guardamos el gris |
| **asociación** | qué detección es de qué roca | **ellos** — global, no greedy |

Corolario de diseño: replicarlos literal **importa su pérdida**. Lo que vale la
pena robarles es la asociación, alimentada con nuestro gris.

## 3. Lo implementado

```
debug/deto1_mascara.py   video → diff frame a frame → binario (histéresis)
                                → detecciones por frame (componentes conexas)
debug/deto1_flujo.py     detecciones → asociación global → trayectorias
debug/deto1_barrido.py   barre el espacio de parámetros y ordena por rectitud
```

Salidas en `debug/out/8_deto1/`. La etapa 1 es la cara (una lectura del video) y
guarda **permisivo** a propósito: el filtro de forma vive en la etapa 2, que es
barata de repetir.

**Asociación global**: DP greedy de caminos más cortos sucesivos sobre el DAG
temporal (Pirsiavash et al., 2011). Se extrae el mejor camino, se marcan sus
detecciones y se repite mientras el camino tenga costo negativo. El min-cost
flow exacto (OR-Tools, ya instalado) queda para después: el greedy da el mismo
tipo de resultado y se puede cortar en cualquier momento.

## 4. Números de la primera corrida completa

| | |
|---|---|
| clip | 3160-789, 453 frames, 3840×2160 |
| etapa 1 | **1.255.125 detecciones**, 182 s |
| binario acumulado | cubre **46,7%** del cuadro (umbral 10/20 σ) |
| tras filtro de forma | 55.121 detecciones (elong≥4, área≥25) = 136/frame |
| arcos del grafo | 4.982.981 |
| etapa 2 | 149 trayectorias en 22 s |

## 5. Los tres errores que costaron la noche, y su lección

**a) `np.nonzero(lab == i)` por componente.** A 4K eso recorre 8,3 millones de
píxeles *una vez por componente*: con miles por frame, el script no terminó un
solo frame en 15 minutos y no daba señales de vida. Se arregla midiendo cada
componente dentro de su propio bounding box. → *A 4K, cualquier operación
por-componente sobre el cuadro completo es una bomba.*

**b) Premiar cada detección sin cobrar la incoherencia.** Primera versión: 154
trayectorias con **mediana de 268 detecciones** sobre un clip de 405 frames, es
decir «rocas» volando nueve segundos. El DP encadenaba manchas de polvo vecinas
porque cada detección daba premio y ningún enlace corto dolía. → *El enlace
mediocre tiene que costar más que el premio, o alargar siempre conviene.*

**c) `|cos|` deja rebotar.** Una estela es un segmento y no distingue ida de
vuelta, así que avanzar 20 px al este y volver 20 al oeste puntuaba igual de
bien: el camino oscilaba sobre el mismo grupo de manchas. Se arregla metiendo la
**dirección en el estado del DP** (16 sectores de 22,5°, tolerancia ±1). Con
tolerancia ±3 las culebras vuelven — está medido. → *Sin memoria de dirección,
no hay criterio que impida el rebote.*

También quedó claro que **penalizar la distancia recorrida es penalizar a las
rocas rápidas**, que son las peligrosas. Lo que debe costar es la incoherencia:
doblar, o que el salto no calce con el largo de la estela.

**La estela mide la velocidad.** El rastro de un frame es lo que la roca se
movió durante la exposición, así que el desplazamiento por frame tiene que
parecerse al largo de la estela. Es una restricción física de primer orden y es
la que separa una roca (estela larga, salto grande) de una mancha de humo
(estela corta, salto chico) sin necesitar un grafo de segundo orden.

## 6. Dónde está parado esto

`trayectorias.png` de la primera corrida es **un ovillo dentro de la nube de
polvo**, no el abanico radial de estelas que se ve en la máscara del cliente
(`debug/mascara_cambios_final_sinbin_7.png`). El pipeline funciona; lo que
encuentra todavía no son rocas.

La métrica que lo delata sin mirar la imagen es la **rectitud** (neto/recorrido):
un vuelo balístico da ~0,9 y una culebra ~0,2. Subir la velocidad mínima de 15 a
30 px/frame ya la sube a **0,82**, lo que apunta a que el punto de operación
—y no el planteamiento— es buena parte del problema.

## 6bis. Cómo se juzga una corrida (y por qué no basta la rectitud)

`barrido.csv` va ordenado por **score**, no por rectitud. Motivo: una trayectoria
de 4 detecciones es recta *por construcción* y marca 1,00, así que ordenar por
rectitud a secas premia justo a las configuraciones que no encontraron nada
(medido: `elong=6` produce 1 trayectoria de 4 puntos con rectitud 1,00).

    rectitud     media ponderada por largo (una traza de 40 puntos pesa 10x
                 más que una de 4)
    rect_largas  rectitud solo de las trayectorias con >= 10 detecciones
    score        largas x rect_largas — encontrar mucho y torcido puntúa tan
                 mal como encontrar poco y recto

Otro tropiezo que dejó lección: el barrido murió sin completar **una sola**
combinación porque empezaba por la más densa (`elong=3` deja 151.217 detecciones
≈ 37 M de arcos) y esta máquina tiene ~1 GB libre. Ahora estima los arcos por
muestreo antes de construir el grafo, baja el `gap` hasta que quepa y recorre las
combinaciones de liviana a pesada, para que un corte a media noche deje igual
resultados utilizables.

## 6ter. Resultado del barrido (54 corridas, 20 min)

Mejor combinación por score: **elong=3, dmin=15, entrada=1.0, tol_sector=1**
(gap bajado a 2 por el tope de arcos) → 2.788 trayectorias, **365 largas** con
rectitud **0,82**. Tabla completa en `barrido.csv`.

Lo que enseña la tabla:

- **`tol_sector` es el control decisivo.** Con 1, la rectitud de las largas
  queda en 0,81–0,85; con 2 se desploma a 0,60–0,62 con el mismo material. La
  memoria de dirección es lo que separa una trayectoria de una culebra.
- `dmin` alto limpia pero empobrece: 30 px/frame sube la rectitud a 0,85 y deja
  108 largas, contra 365 con dmin=15.
- `entrada=2.5` mata casi todo: exige ~7 detecciones netas para abrir una
  trayectoria y casi nada llega.

## 6quater. El diagnóstico de fondo: lo que encuentra es POLVO

Comparadas lado a lado, la máscara del cliente muestra **estelas radiales largas
que salen del área y cruzan el cuadro** — algunas curvas, de largo alcance, que
son las peligrosas. Nuestro resultado concentra todo **dentro** del área de
tronadura.

Se aisló la periferia para verlo limpio: solo las detecciones a más de 500 px
fuera del hull de la malla (358.953 de 1.255.125). La rectitud sube a **0,89**
—el mejor número de toda la noche— y aun así lo que dibuja es **una banda curva
y densa: el frente de la nube de polvo expandiéndose**, no rocas.

**Ahí está el problema, y no es la asociación.** El frente de polvo cumple
*todos* los criterios cinemáticos que se le exigieron a una roca: se mueve
rápido, es alargado, y avanza coherente en dirección. Ningún costo de enlace lo
distingue, porque por movimiento no se distingue. Es la misma conclusión a la
que se había llegado por otro camino (`PREPROCESO.md` §1: roca y polvo se
traslapan en intensidad, velocidad y suavidad).

Corolario para la pregunta original: **replicar deto1 no nos daría el salto.**
Su cuello de botella y el nuestro es el mismo —separar roca de polvo en la
entrada— y probablemente por eso ellos terminaban pintando a mano.

## 7. Pendiente

- **Leer el barrido** (`barrido.csv`, ordenado por rectitud) y mirar `galeria/`.
- **Confirmar con el cliente si «pintaban» a mano o lo hacía su algoritmo.** Si
  era a mano, no hay nada que replicar ahí: esa herramienta ya la tenemos en la
  vista. Si era automático, es reconstrucción y es la parte interesante.
- Alimentar la misma asociación con **nuestro gris** en vez del binario, y medir
  la diferencia. Es la comparación que contesta si su camino nos conviene.
- Restricción radial desde la **línea de tiros** (no desde un punto: eso ya lo
  descartó el cliente, ver `flyrocks-denoising`).
- Min-cost flow exacto con OR-Tools, si el greedy muestra que vale la pena.
