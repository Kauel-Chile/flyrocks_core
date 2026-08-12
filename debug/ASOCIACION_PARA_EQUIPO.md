# Asociación trayectoria → tiro de origen

**Resumen del approach para compartir con el equipo.** Autocontenido: no hace
falta leer nada más para entenderlo ni para reimplementarlo.
Fecha: 2026-08-07.

---

## 1. El problema

Tenemos trazos de flyrock sobre el video del dron. El cliente no necesita el
trazo: necesita saber **de qué pozo salió cada roca**. Su versión cruza la malla
de detonación con los tiempos de secuencia y asocia por posición, velocidad y
tiempo.

Lo que hicimos nosotros ataca la **mitad geométrica** del problema, y está pensado
para que la mitad temporal se enchufe encima sin rehacer nada.

---

## 2. Los dos números de los que cuelga todo el diseño

Medidos sobre la malla real de este video (113 pozos, tiempos de detonación todos
únicos, ventana de 3000–5769 ms):

| Dato | Valor |
|---|---|
| Separación mediana entre pozos vecinos | **5.45 m = 47 px** |
| Δt mediano entre esos mismos pozos vecinos | **216 ms = 6.5 frames** (a 29.97 fps) |
| Pares que detonan dentro de 1 frame | 148 de 6328, y están a **19.7 m** |

**Geometría y tiempo son complementarios casi perfectos.** Los pozos que el
espacio no distingue detonan bien separados en el tiempo; los que el tiempo no
distingue están lejos en el espacio. Cada fuente cubre el punto ciego de la otra.

Consecuencia directa de diseño: **la capa geométrica sola no puede dar el pozo,
solo candidatos.** No es un defecto del algoritmo — es información que no está en
la imagen. Y al revés: la capa temporal sola tampoco basta. Solo la combinación
cierra.

---

## 3. Lo que hicimos: asociación geométrica (sin tiempo)

### 3.1 El insumo

Cada trayectoria es una **Bézier cuadrática** (3 puntos: `p0` inicio, `c1`
control, `p2` impacto) en coordenadas de imagen. En esta primera versión se
dibujan a mano sobre la máscara acumulada o sobre un frame previo a la tronadura.

Se dibuja a mano **a propósito**: con trazos manuales el error de detección es
cero, así que si el match falla el culpable es la matemática de asociación y nada
más. Es el experimento limpio que valida la capa antes de conectarle el
detector.

### 3.2 La malla en píxeles

Se proyectan los 113 pozos del CSV de secuencia a píxeles con la homografía ya
calibrada (afín, RMS 3.19 px). Se colorean por `DetonatingTime` con una rampa
secuencial de un solo tono: de una se ve la propagación derecha→izquierda y el
cliente reconoce su propia secuencia. Hay ajuste fino (mover/rotar/escalar) por
si el frame no es el de la calibración.

### 3.3 Corrección de paralaje — el paso que no es obvio

**El inicio visible de un trazo NO está sobre el pozo.** La roca se hace visible
cuando ya subió unos metros, y una cámara cenital proyecta un objeto elevado
**más lejos del nadir** de lo que realmente está. Sesgo sistemático, siempre en la
misma dirección: hacia afuera.

La corrección es:

```
O = nadir + (p0 − nadir) / (1 + k)        con   k = z / (h − z)
```

donde `h` = altura del dron y `z` = altura de la roca al hacerse visible.

Lo importante: **`h` y `z` colapsan en un solo escalar adimensional `k`**. No hay
que conocer ninguno de los dos por separado. Rango útil ≈ 0.02–0.10; a 100 m del
nadir eso vale 2–8 m de corrección, o sea entre medio pozo y pozo y medio.

> **Decidimos NO usar el `.SRT` del dron.** Su altura es relativa al punto de
> despegue; en un rajo abierto, con el operador parado en otra cota que la
> tronadura, ese número miente por decenas de metros. `k` se calibra con un
> slider mirando plausibilidad.

**`k` es el parámetro más sensible del sistema.** Matriz medida de *paralaje real*
contra *paralaje asumido* (acierto top-1):

| | k asumido 0.00 | 0.04 | 0.08 | 0.15 | 0.25 |
|---|---|---|---|---|---|
| **k real 0.00** | **88 %** | 78 % | 46 % | 26 % | 19 % |
| **k real 0.08** | 55 % | 83 % | **88 %** | 56 % | 25 % |
| **k real 0.25** | 19 % | 23 % | 27 % | 52 % | **88 %** |

La diagonal siempre da 88 %; cada paso equivocado cuesta entre 8 y 40 puntos. Si
montas tu propia versión, esto es lo primero que hay que calibrar.

### 3.4 El asociador: una cuña de retroceso

Por cada trayectoria:

1. **Origen corregido** `O` — el de arriba.
2. **Dirección de retroceso** — opuesta a la tangente inicial de la Bézier,
   o sea el vector `c1 → p0` normalizado. No se extrapola la curva hacia atrás:
   se prolonga **recta**. Justificado por medición: para retrocesos cortos
   (~20 m) la curva real se desvía 0.8–2.3 m de la recta, menos que la separación
   entre pozos. A 80 m se desvía 8–25 m y deja de valer.
3. **Costo por pozo.** Con `a` = avance hacia atrás (proyección sobre el eje) y
   `perp` = desvío perpendicular al eje:

   | Término | Regla | Por qué |
   |---|---|---|
   | Causalidad | pozo *adelante* del inicio (`a < −8 m`) → descartar | la roca no viaja hacia atrás |
   | Alineamiento | `perp / (tan σ · max(a, 15 m))` | la tolerancia **se abre con la distancia**: es una cuña, no un círculo |
   | Lejanía | `a / 70 m` | la roca se hace visible cerca del pozo, no a 80 m |

   `costo = alineamiento² + lejanía²`

4. **Ranking:** los 5 mejores, con confianza relativa por **softmax** del costo.

Pseudocódigo completo (es literalmente todo el algoritmo):

```js
O   = nadir + (p0 − nadir) / (1 + k)
dir = normalize(p0 − c1)                    // hacia atrás

para cada pozo q:
    v    = q − O
    a    = dot(v, dir)                      // avance hacia atrás
    si a < −8 m: descartar                  // causalidad
    perp = |cross(v, dir)|                  // desvío del eje
    ang  = perp / (tan(σ) · max(a, 15 m))
    lej  = max(0, a) / 70 m
    costo = ang² + lej²

ordenar por costo, tomar 5
conf_i = softmax(−costo_i / 2)
```

**σ = 4° confirmado por medición** (con 2° el acierto cae a 65 %, con 20° a 56 %).

### 3.5 Salida: candidatos con degradación honesta

Nunca se reporta un pozo único. Tres niveles según cuánto destaca el mejor
candidato:

- **`origen:`** — confianza ≥ 0.55 → el pozo
- **`origen probable:`** — ≥ 0.28 → un grupo
- **`zona:`** — por debajo → un sector

Es más defendible frente al cliente que un pozo único que él puede desmentir
señalando el cráter de al lado.

Además, a pedido explícito del cliente (*"que la trayectoria salga y se dibuje
desde el pozo más plausible"*), se dibuja un **tramo de empalme** del pozo al
inicio del trazo: Bézier cuadrática cuyo punto de control se apoya en la **misma
tangente** con que arranca la curva dibujada, así el empalme no tiene quiebre y se
lee como una sola trayectoria. Mismo color pero **punteado** — dice "este tramo no
se ve en el video, lo puso el sistema" sin aparentar certeza que no hay.

---

## 4. Qué tan bien funciona

Banco de pruebas: 113 pozos × repeticiones, trayectorias sintéticas con origen
conocido, variando cuánto tapa el humo el arranque y cuánto se equivoca el trazo.

**Con `k` bien calibrado: top-1 = 88 %, top-3 = 99 %.**
El pozo verdadero **nunca salió del top-5** en ninguna corrida.

| | 0° err | 2° err | 5° err | 10° err |
|---|---|---|---|---|
| top-1 | 88 % | 86 % | 78 % | 73 % |
| top-3 | 100 % | 100 % | 99 % | 97 % |

**El modo de falla, caracterizado:** cuando se equivoca, el **67 % de las veces
elige un pozo más externo** que el verdadero, a 8.0 m de mediana (≈1.5 pozos). La
causa es estructural — los pozos **alineados sobre el eje de vuelo** son
indistinguibles con geometría sola, y el algoritmo apuesta por el más cercano al
inicio del trazo.

Y esos pozos alineados detonan separados por 216 ms. **El tiempo lo resuelve de
raíz.**

**Lo que NO está verificado:** está medido contra trayectorias sintéticas, no
contra trazos reales con origen conocido. `k` sigue sin calibrar con datos reales.

---

## 5. Lo que viene: sumar la dimensión temporal

Acá es donde tu trabajo con trayectorias temporales se enchufa. La idea es que el
asociador **no cambia** — se le agregan restricciones.

### 5.1 El puente: herencia temporal

Problema: la máscara acumulada no tiene eje temporal, así que un trazo dibujado a
mano no trae `t`.

Solución: las detecciones del pipeline sí traen `[id, x, y, t]`. Una curva
dibujada puede **heredar el tiempo** de las detecciones que caen dentro de su
corredor. El discriminador correcto no es el simple contacto sino la **fracción
del largo** de la detección que cae dentro del trazo.

*(Si tú trabajas directo sobre detecciones temporales, este paso te lo saltas
entero — ya tienes `t`.)*

### 5.2 Las dos restricciones nuevas

Con `t` disponible el problema pasa a estar **sobredeterminado**: 4 observaciones
contra 1 incógnita.

1. **Causalidad dura** — el pozo debe haber detonado **antes** de que nazca la
   traza. No tiene ningún parámetro que calibrar: elimina candidatos gratis.
2. **Tiempo de vuelo coherente** — `t_nacimiento − t_detonación` debe ser corto y
   consistente. Como los pozos vecinos están separados por 6.5 frames, **esta
   ventana sola deja 2–3 candidatos**, y la geometría desempata.

### 5.3 Refinamiento por punto fijo

Con origen + impacto + tiempo de vuelo, la balística queda determinada → estimas
la **altura real** de la roca → corriges el paralaje con un `k` **propio de cada
traza** en vez de uno global → reasocias. Converge en 2–3 vueltas.

Esto cierra el círculo: el parámetro más sensible del sistema deja de ser un
slider y pasa a salir de los datos.

### 5.4 Dos advertencias para no equivocar la implementación

- **NO es asignación 1-a-1.** Un pozo lanza varias rocas. Hungarian puro está
  mal: hay que usar **min-cost flow con capacidad** en el pozo.
- **El eslabón débil es cuándo "nace" una traza.** El destello y el humo tapan el
  origen, y ese retardo determina todo el cálculo de tiempo de vuelo. **Hay que
  medirlo, no asumirlo** — y se mide justamente con los trazos manuales de la
  fase geométrica cuyo pozo ya validamos.

  → **La fase geométrica calibra la fase temporal.** No son dos caminos
  paralelos: el primero produce el ground truth del segundo.

---

## 6. Un detalle de geometría que conviene saber antes de programar

**Una parábola balística vista desde el dron NO se ve como una parábola en la
imagen.** Proyectando la balística por una cámara pinhole cenital sale:

```
r_img(t) ∝ (r0 + v·t) / (h − u·t + ½g·t²)
```

El denominador (la distancia a la cámara) cambia durante el vuelo, y eso deforma
la cónica. Consecuencias prácticas:

- **La traza puede devolverse en la imagen.** Punto de retroceso en
  `t* = √(2h/g)`, y ocurre dentro del vuelo si `h < 4·z_max`. Con el dron a
  ~250 m, toda roca que suba de ~62 m retrocede en la imagen. No porque la roca
  vuelva: al caer se aleja de la cámara y se encoge más rápido de lo que avanza.
  **Si tu tracker filtra por continuidad sin reversas, esto te parte las
  trayectorias largas en dos pedazos.** Lo estamos viendo.
- **Solo el origen y el impacto son convertibles a metros con la homografía**,
  porque están en `z ≈ 0`. Los puntos intermedios del arco están en el aire y
  `H⁻¹` los manda a una posición equivocada. Ojo al medir distancias de alcance.

---

## 7. Si quieres comparar approaches

El contrato mínimo para que los dos lados sean comparables:

**Entrada por trayectoria:**
```json
{ "id": "...", "fuente": "manual|pipeline",
  "p0": [x, y], "c1": [x, y], "p2": [x, y],
  "t_inicio": null, "t_fin": null }
```
(coordenadas en píxeles de imagen; `t` en frames o ms, opcional)

**Salida por trayectoria:**
```json
{ "id": "...", "nivel": "pozo|grupo|sector",
  "candidatos": [ {"pozo": 47, "conf": 0.62}, ... ] }
```

Con eso, la métrica de comparación es directa: **top-1 y top-3 sobre el mismo set
de trazos**, y la distancia en metros del pozo elegido al verdadero cuando falla.

Lo más útil que puede salir de comparar: si tu versión temporal y esta geométrica
fallan en **trazos distintos**, entonces la combinación va a superar a las dos por
separado — que es exactamente lo que predice el §2. Si fallan en los mismos, hay
un problema común aguas arriba (probablemente la homografía o el nacimiento de la
traza).
