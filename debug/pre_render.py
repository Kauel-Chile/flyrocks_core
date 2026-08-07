"""
PASOS 1, 3 y 4 — renders y comparaciones desde el cache de pre_pasada.py.
Instantaneo (no vuelve a leer el video).

  Paso 1  00_baseline/  la mascara gris actual, congelada como referencia
  Paso 3  02_zscore/    la misma diferencia medida en SIGMAS locales
  Paso 4  01_fondo/     diff contra fondo ("pantalla verde") vs frame-a-frame

    uv run python debug/pre_render.py
"""
import cv2
import numpy as np

from pre_common import (D_BASELINE, D_ZSCORE, D_FONDO, NOISE_THRESHOLD,
                        ensure_dirs, to8, falsecolor, side_by_side, grid, save)
import pre_pasada

# recorte de detalle: en 4K reducido a pantalla, lo tenue desaparece
CROP_W, CROP_H = 1500, 950


def auto_crop(acc, thr):
    """Ventana centrada en el centroide de la actividad fuerte."""
    m = acc > thr
    ys, xs = np.nonzero(m)
    if len(xs) < 100:
        cy, cx = acc.shape[0] // 2, acc.shape[1] // 2
    else:
        cy, cx = int(np.median(ys)), int(np.median(xs))
    h, w = acc.shape
    x0 = int(np.clip(cx - CROP_W // 2, 0, w - CROP_W))
    y0 = int(np.clip(cy - CROP_H // 2, 0, h - CROP_H))
    return x0, y0


def cut(img, box):
    x0, y0 = box
    return img[y0:y0 + CROP_H, x0:x0 + CROP_W]


def stats(name, mask):
    pct = 100.0 * mask.mean()
    print(f"    {name:<34} {int(mask.sum()):>10,} px activos  ({pct:5.2f}% del cuadro)")


def main():
    ensure_dirs()
    d = pre_pasada.load()
    acc_ff, acc_zff = d["acc_ff"], d["acc_zff"]
    acc_bg, acc_zbg = d["acc_bg"], d["acc_zbg"]

    box = auto_crop(acc_ff, NOISE_THRESHOLD * 3)
    print(f"Recorte de detalle en x={box[0]} y={box[1]} ({CROP_W}x{CROP_H})\n")

    # ---------------- PASO 1: baseline congelado ----------------------
    print("PASO 1 — baseline (lo que el pipeline ve hoy)")
    base_gray = to8(acc_ff, hi_pct=99.5)
    save(D_BASELINE / "baseline_gris.png", base_gray)
    bin_base = (acc_ff > NOISE_THRESHOLD)
    save(D_BASELINE / "baseline_umbral8.png",
         (bin_base * 255).astype(np.uint8))
    save(D_BASELINE / "baseline_crop.png", cut(base_gray, box))
    stats(f"baseline  absdiff > {NOISE_THRESHOLD}", bin_base)

    # ---------------- PASO 3: z-score --------------------------------
    print("\nPASO 3 — mismo diff, medido en sigmas locales")
    z_gray = to8(acc_zff, hi_pct=99.5)
    save(D_ZSCORE / "zscore_gris.png", z_gray)
    save(D_ZSCORE / "zscore_crop.png", cut(z_gray, box))

    items = []
    for k in (3, 4, 5, 6, 8, 12):
        m = acc_zff > k
        stats(f"z-score  > {k} sigmas", m)
        items.append(((m * 255).astype(np.uint8), f"z > {k} sigmas"))
    items.append((((bin_base) * 255).astype(np.uint8),
                  f"BASELINE absdiff > {NOISE_THRESHOLD}"))
    save(D_ZSCORE / "zscore_barrido_umbral.png",
         grid([(cut(i, box), t) for i, t in items], cols=4))

    save(D_ZSCORE / "comparacion_baseline_vs_zscore.png", side_by_side([
        (cut(base_gray, box), "PASO 1 baseline: intensidad absoluta"),
        (cut(z_gray, box), "PASO 3: sigmas locales (umbral adaptativo)"),
    ]))

    # ---------------- PASO 4: diff contra fondo -----------------------
    print("\nPASO 4 — diff contra fondo (pantalla verde) vs frame-a-frame")
    bg_gray = to8(acc_bg, hi_pct=99.5)
    zbg_gray = to8(acc_zbg, hi_pct=99.5)
    save(D_FONDO / "difffondo_gris.png", bg_gray)
    save(D_FONDO / "difffondo_zscore_gris.png", zbg_gray)
    save(D_FONDO / "difffondo_crop.png", cut(bg_gray, box))
    for k in (8, 16, 24):
        stats(f"diff-contra-fondo > {k}", acc_bg > k)

    save(D_FONDO / "comparacion_ff_vs_fondo.png", side_by_side([
        (cut(base_gray, box), "frame-a-frame (mide CAMBIO)"),
        (cut(bg_gray, box), "contra fondo (mide PRESENCIA)"),
    ]))

    # ---------------- las cuatro capas juntas -------------------------
    save(D_BASELINE / "comparacion_4_capas.png", grid([
        (cut(base_gray, box), "1. baseline: |curr-prev|"),
        (cut(z_gray, box), "3. z-score: |curr-prev| / sigma"),
        (cut(bg_gray, box), "4. contra fondo: |curr-fondo|"),
        (cut(zbg_gray, box), "4b. contra fondo en sigmas"),
    ], cols=2))

    # vista global (sin recorte) de las dos candidatas principales
    save(D_BASELINE / "comparacion_global.png", side_by_side([
        (base_gray, "BASELINE (hoy)"),
        (z_gray, "Z-SCORE (umbral adaptativo)"),
        (bg_gray, "CONTRA FONDO"),
    ]))
    print()


if __name__ == "__main__":
    main()
