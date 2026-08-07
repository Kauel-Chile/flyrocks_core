"""
PASO 5 — Realce de LINEALIDAD (cambiar el eje: de "cuanto brilla" a "cuanto se
parece a una linea").

Idea: roca = segmento lineal fino y orientado; polvo = mancha difusa sin
orientacion. Intensidad, velocidad y persistencia se traslapan entre ambos
(ver ESTADO_Y_PENDIENTES.md §4); la FORMA no.

Metodo: banco de filtros ORIENTADOS.
  resp[t] = convolucion con un kernel de LINEA en la orientacion t
  Se usa convolucion (no apertura morfologica) a proposito: las estelas salen
  PUNTEADAS (la roca aparece intermitente frame a frame) y la convolucion
  INTEGRA a lo largo de la direccion, uniendo los puntos alineados. Una apertura
  morfologica las rompe.

  lineness = max_t(resp) - mediana_t(resp)      <- ANISOTROPIA

Por que la resta: una mancha responde IGUAL en todas las orientaciones (max ~
mediana -> lineness ~ 0). Una linea responde fuerte solo en su orientacion
(max >> mediana -> lineness alto). Asi una estela TENUE supera a una mancha
BRILLANTE, que es justo lo que el umbral de intensidad no logra.

Salidas en 7_preproceso/03_linealidad/.

    uv run python debug/pre_linealidad.py [--len 31] [--nang 16]
"""
import argparse
import time

import cv2
import numpy as np

from pre_common import (D_LINEAL, ensure_dirs, to8, side_by_side, grid, save)
import pre_pasada
import pre_render


def line_kernel(length, angle_deg, thickness=1.0):
    """Kernel de linea normalizado (media a lo largo de la direccion angle)."""
    s = int(length) | 1                       # impar
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
    """Kernel de disco normalizado (referencia ISOTROPICA)."""
    s = int(diameter) | 1
    k = np.zeros((s, s), np.float32)
    cv2.circle(k, (s // 2, s // 2), s // 2, 1.0, -1)
    return k / k.sum()


def lineness(img, length=31, nang=16, mode="disco"):
    """Devuelve (lineness, orientacion_deg, respuesta_maxima).

    mode="mediana": max_t - mediana_t. Simple, pero un punto MUY brillante deja
        un artefacto en estrella (responde en todas las orientaciones y el
        aliasing de los kernels impide que la mediana lo cancele).
    mode="disco":  max_t - respuesta a un DISCO del mismo tamano. Un punto o una
        mancha responden fuerte al disco y se cancelan; una linea responde debil
        al disco (el disco promedia mucha area vacia alrededor) y sobrevive.
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


def orientation_view(lin, ori, hi_pct=99.5):
    """HSV: tono = direccion de la linea, brillo = cuan lineal es."""
    v = to8(lin, hi_pct=hi_pct)
    hsv = np.zeros(lin.shape + (3,), np.uint8)
    hsv[..., 0] = (ori / 2.0).astype(np.uint8)      # 0..180 -> 0..90 (OpenCV hue)
    hsv[..., 1] = 255
    hsv[..., 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main(length, nang):
    ensure_dirs()
    d = pre_pasada.load()
    acc_ff = d["acc_ff"]
    acc_zff = d["acc_zff"]

    box = pre_render.auto_crop(acc_ff, 24)
    cut = lambda im: pre_render.cut(im, box)      # noqa: E731

    print(f"Banco orientado: largo={length}px, {nang} orientaciones\n")

    t0 = time.time()
    lin_ff, ori_ff, _ = lineness(acc_ff, length, nang, mode="disco")
    print(f"  sobre BASELINE   ({time.time() - t0:.0f}s)")
    t0 = time.time()
    lin_z, ori_z, _ = lineness(acc_zff, length, nang, mode="disco")
    print(f"  sobre Z-SCORE    ({time.time() - t0:.0f}s)")

    base8 = to8(acc_ff, hi_pct=99.5)
    lin8 = to8(lin_ff, hi_pct=99.8)
    linz8 = to8(lin_z, hi_pct=99.8)

    # --- variantes: como se suprime el fondo, y a que escala ----------
    lin_med, _, _ = lineness(acc_ff, length, nang, mode="mediana")
    save(D_LINEAL / "variantes_supresion.png", side_by_side([
        (cut(to8(lin_med, hi_pct=99.8)), "max - mediana (deja estrellas)"),
        (cut(lin8), "max - disco (suprime puntos/manchas)"),
    ]))

    escalas = []
    for L in (15, 31, 51, 81):
        li, _, _ = lineness(acc_ff, L, nang, mode="disco")
        escalas.append((cut(to8(li, hi_pct=99.8)), f"kernel largo = {L} px"))
        print(f"  escala L={L} lista")
    save(D_LINEAL / "variantes_escala.png", grid(escalas, cols=2))

    save(D_LINEAL / "linealidad_gris.png", lin8)
    save(D_LINEAL / "linealidad_desde_zscore.png", linz8)
    save(D_LINEAL / "linealidad_crop.png", cut(lin8))
    save(D_LINEAL / "linealidad_orientacion.png",
         orientation_view(lin_ff, ori_ff))
    save(D_LINEAL / "linealidad_orientacion_crop.png",
         cut(orientation_view(lin_ff, ori_ff)))

    # --- la comparacion que importa -----------------------------------
    save(D_LINEAL / "comparacion_intensidad_vs_linealidad.png", side_by_side([
        (cut(base8), "PASO 1 — intensidad (lo que hay hoy)"),
        (cut(lin8), "PASO 5 — LINEALIDAD (linea vs mancha)"),
    ]))
    save(D_LINEAL / "comparacion_global.png", side_by_side([
        (base8, "intensidad"),
        (lin8, "linealidad"),
        (linz8, "linealidad sobre z-score"),
    ]))

    # --- cuanto separa? el test honesto -------------------------------
    # Se compara el TOP 1% de cada mascara: cuanto de eso es estructura fina.
    print("\n  Fraccion del cuadro sobre umbral (barrido por percentil):")
    items = []
    for p in (90, 95, 98, 99, 99.5):
        ti = np.percentile(acc_ff, p)
        tl = np.percentile(lin_ff, p)
        mi = (acc_ff > ti)
        ml = (lin_ff > tl)
        items.append(((cut(ml) * 255).astype(np.uint8), f"linealidad p{p}"))
        print(f"    p{p:<5} intensidad>{ti:6.1f}   linealidad>{tl:6.2f}")
    save(D_LINEAL / "linealidad_barrido_percentil.png", grid(items, cols=3))

    np.savez_compressed(D_LINEAL.parent / "_cache" / "linealidad.npz",
                        lin_ff=lin_ff.astype(np.float32),
                        ori_ff=ori_ff.astype(np.float32),
                        lin_z=lin_z.astype(np.float32))
    print(f"\n  cache -> _cache/linealidad.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 51 px elegido tras el barrido de escala (variantes_escala.png):
    #   15 -> domina la textura fina del terreno
    #   51 -> las estelas largas y las curvas dominan  <-- mejor balance
    #   81 -> el humo se "peina" en filamentos falsos
    ap.add_argument("--len", type=int, default=51, dest="length")
    ap.add_argument("--nang", type=int, default=16)
    a = ap.parse_args()
    main(a.length, a.nang)
