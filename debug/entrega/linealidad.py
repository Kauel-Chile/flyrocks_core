"""
REALCE DE LINEALIDAD — version autocontenida (un solo archivo).

Que hace
--------
Cambia el eje con el que se mira una mascara: de "cuanto brilla cada pixel" a
"cuanto se parece a una LINEA lo que hay en cada pixel".

En cada punto de la imagen se prueba una "reglita" de N pixeles en 16
direcciones distintas y se suma el brillo que hay debajo en cada una. Se
conserva la direccion que mas sumo, y a ese valor se le resta lo que suma un
DISCO del mismo diametro en el mismo punto:

    linealidad = max_sobre_orientaciones(respuesta) - respuesta_del_disco

Si en ese punto hay una raya, la reglita alineada con ella suma mucho y el
disco suma poco (abarca el vacio de alrededor) -> la resta da un valor alto.
Si hay una mancha de humo, ambos suman parecido -> la resta da casi cero.

Resultado: una estela TENUE le gana a una mancha BRILLANTE, que es justo lo
que un umbral de intensidad no logra.

Por que convolucion y no apertura morfologica
---------------------------------------------
Las estelas salen PUNTEADAS (la roca aparece intermitente de frame a frame).
La convolucion INTEGRA a lo largo de la direccion y une los puntos alineados;
una apertura morfologica los rompe.


Entradas y salidas
------------------
ENTRA:  UNA imagen, en escala de grises, sin binarizar. Tipicamente la mascara
        de intensidad acumulada (max de los diffs a lo largo del video). No usa
        el video, no usa el tiempo: es un proceso puramente espacial sobre una
        imagen fija.

SALE:   - PNG de la linealidad (normalizado por percentil, solo para VER).
        - opcional --npy: los valores CRUDOS en float32 (los que sirven para
          umbralizar aguas abajo).
        - opcional --orientacion: vista en color donde el TONO es la direccion
          de la linea y el BRILLO es cuan lineal es.


Uso
---
    python linealidad.py mascara_intensidad.png
    python linealidad.py mascara.png -o lin.png --npy lin.npy --orientacion ori.png
    python linealidad.py mascara.png --len 31 --nang 24

Como libreria:
    from linealidad import lineness
    lin, ori, mx = lineness(img_gris_float, length=51, nang=16, mode="disco")


OJO CON EL PARAMETRO --len (es el unico que hay que calibrar)
-------------------------------------------------------------
Es el largo de la reglita EN PIXELES, asi que depende de la resolucion y de
que tan largas se vean las estelas. El valor por defecto (51) se calibro sobre
video 4K (3840x2160). Barrido observado a esa resolucion:

    15 px -> domina la textura fina del terreno (mucho falso positivo)
    51 px -> las estelas largas y las curvas dominan   <-- mejor balance
    81 px -> el humo se "peina" en filamentos falsos

Si tu imagen es de otra resolucion, escala el valor proporcionalmente
(p.ej. en 1080p, la mitad: ~25).


IMPORTANTE SOBRE LA SALIDA
--------------------------
Los valores de la linealidad NO estan normalizados: su rango depende de la
imagen de entrada. Para binarizar hay que cortar POR PERCENTIL, nunca por un
valor absoluto fijo. El PNG que se guarda ya viene estirado por percentil para
que se vea; para calcular, usa el .npy.

Dependencias: numpy, opencv-python. Nada mas.
Costo: ~3 s sobre una imagen 4K (16 convoluciones + 1 disco).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# KERNELS
# ----------------------------------------------------------------------
def line_kernel(length, angle_deg, thickness=1.0):
    """Kernel de LINEA normalizado: promedia a lo largo de la direccion dada.

    Se rasteriza el segmento sobremuestreado (s*4 pasos) para que la linea
    quede continua a cualquier angulo y no con huecos por redondeo.
    """
    s = int(length) | 1                       # tamano impar (centro definido)
    k = np.zeros((s, s), np.float32)
    c = s // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    for t in np.linspace(-c, c, s * 4):
        x = int(round(c + dx * t))
        y = int(round(c + dy * t))
        if 0 <= x < s and 0 <= y < s:
            k[y, x] = 1.0
    if thickness > 1.0:
        k = cv2.GaussianBlur(k, (0, 0), thickness / 2.0)
    tot = k.sum()
    return k / tot if tot > 0 else k


def disk_kernel(diameter):
    """Kernel de DISCO normalizado: la referencia ISOTROPICA (sin direccion)."""
    s = int(diameter) | 1
    k = np.zeros((s, s), np.float32)
    cv2.circle(k, (s // 2, s // 2), s // 2, 1.0, -1)
    return k / k.sum()


# ----------------------------------------------------------------------
# EL FILTRO
# ----------------------------------------------------------------------
def lineness(img, length=51, nang=16, mode="disco"):
    """Realce de linealidad. Devuelve (linealidad, orientacion_deg, respuesta_max).

    img    : imagen 2D en escala de grises, SIN binarizar (cualquier rango).
    length : largo de la reglita en px. Ver la nota de calibracion arriba.
    nang   : cuantas orientaciones se prueban entre 0 y 180 grados.
    mode   : como se suprime el fondo.
        "disco"   : max_orientaciones - respuesta a un DISCO del mismo tamano.
                    Un punto o una mancha responden fuerte al disco y se
                    cancelan; una linea responde debil al disco (el disco
                    promedia mucha area vacia alrededor) y sobrevive.
                    <-- RECOMENDADO
        "mediana" : max_orientaciones - mediana_orientaciones. Mas simple, pero
                    un punto MUY brillante deja un artefacto en ESTRELLA:
                    responde en todas las orientaciones y el aliasing de los
                    kernels impide que la mediana lo cancele del todo.

    Devuelve tres arrays float32 del mismo tamano que la entrada:
        [0] linealidad     >= 0. A mayor valor, mas se parece a una linea.
                           ES LA SALIDA QUE SE USA.
        [1] orientacion    direccion de la linea dominante, en grados 0..180.
        [2] respuesta_max  la respuesta cruda antes de restar el fondo
                           (util solo para diagnostico).

    Memoria: reserva nang * alto * ancho float32. En 4K con nang=16 son ~530 MB.
    Si te quedas corto de RAM, baja nang a 8 o procesa por bloques.
    """
    f = img.astype(np.float32)
    angles = np.linspace(0, 180, nang, endpoint=False)
    resps = np.empty((nang,) + f.shape, np.float32)
    for i, a in enumerate(angles):
        resps[i] = cv2.filter2D(f, cv2.CV_32F, line_kernel(length, a))

    mx = resps.max(axis=0)
    if mode == "disco":
        base = cv2.filter2D(f, cv2.CV_32F, disk_kernel(length))
    else:
        base = np.median(resps, axis=0)

    arg = resps.argmax(axis=0).astype(np.float32)
    ori = arg * (180.0 / nang)
    return np.maximum(mx - base, 0.0), ori, mx


# ----------------------------------------------------------------------
# VISUALIZACION
# ----------------------------------------------------------------------
def to8(img, lo_pct=0.0, hi_pct=99.8):
    """Normaliza a uint8 estirando por PERCENTILES.

    No se usa min/max porque un solo pixel atipico aplastaria todo lo demas.
    """
    f = img.astype(np.float32)
    lo = float(np.percentile(f, lo_pct))
    hi = float(np.percentile(f, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def orientation_view(lin, ori, hi_pct=99.5):
    """Vista en color: TONO = direccion de la linea, BRILLO = cuan lineal es."""
    hsv = np.zeros(lin.shape + (3,), np.uint8)
    hsv[..., 0] = (ori / 2.0).astype(np.uint8)     # 0..180 -> 0..90 (hue OpenCV)
    hsv[..., 1] = 255
    hsv[..., 2] = to8(lin, hi_pct=hi_pct)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Realce de linealidad sobre una mascara en escala de grises.")
    ap.add_argument("entrada", help="PNG/JPG de la mascara (escala de grises)")
    ap.add_argument("-o", "--salida", default=None,
                    help="PNG de salida (por defecto: <entrada>_linealidad.png)")
    ap.add_argument("--npy", default=None,
                    help="guarda ademas los valores CRUDOS float32 (.npy)")
    ap.add_argument("--orientacion", default=None,
                    help="guarda ademas la vista de orientacion en color (PNG)")
    ap.add_argument("--len", type=int, default=51, dest="length",
                    help="largo de la reglita en px (default 51, calibrado en 4K)")
    ap.add_argument("--nang", type=int, default=16,
                    help="numero de orientaciones entre 0 y 180 (default 16)")
    ap.add_argument("--modo", choices=("disco", "mediana"), default="disco",
                    help="supresion del fondo (default disco, recomendado)")
    a = ap.parse_args()

    src = Path(a.entrada)
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"No pude leer la imagen: {src}")
    if img.ndim == 3:
        print("  aviso: la entrada es a color, se convierte a gris")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"  entrada : {src.name}  {img.shape[1]}x{img.shape[0]}  {img.dtype}")
    print(f"  kernel  : largo={a.length}px  orientaciones={a.nang}  modo={a.modo}")

    t0 = time.time()
    lin, ori, _ = lineness(img, a.length, a.nang, mode=a.modo)
    print(f"  listo en {time.time() - t0:.1f}s")

    out = Path(a.salida) if a.salida else src.with_name(src.stem + "_linealidad.png")
    cv2.imwrite(str(out), to8(lin, hi_pct=99.8))
    print(f"  -> {out}")

    if a.npy:
        np.save(a.npy, lin.astype(np.float32))
        print(f"  -> {a.npy}  (valores crudos float32)")
    if a.orientacion:
        cv2.imwrite(str(a.orientacion), orientation_view(lin, ori))
        print(f"  -> {a.orientacion}  (tono = direccion)")

    # Referencia para elegir el corte aguas abajo: SIEMPRE por percentil.
    print("\n  Percentiles de la linealidad (usa estos para binarizar):")
    for p in (90, 95, 98, 99, 99.5):
        print(f"    p{p:<5} > {np.percentile(lin, p):8.3f}")


if __name__ == "__main__":
    main()
