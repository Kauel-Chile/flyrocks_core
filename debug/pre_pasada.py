"""
PASADA UNICA sobre el clip -> alimenta los pasos 1, 3 y 4.

Hace UNA sola lectura del video (la etapa cara) y cachea cuatro acumulados, para
poder iterar visualizaciones y umbrales en milisegundos despues.

Diferencia clave con el pipeline actual: cada frame se registra DIRECTO contra el
fondo (no encadenado), asi no hay deriva acumulada a lo largo de los 15 s.

Acumulados (maximo por pixel sobre todos los frames desde START):
  acc_ff  -> |curr - prev|                     BASELINE (lo que se hace hoy)
  acc_zff -> |curr - prev| / sigma_diff        paso 3: mismo diff, en SIGMAS
  acc_bg  -> |curr - fondo|                    paso 4: "pantalla verde"
  acc_zbg -> |curr - fondo| / sigma_bg         paso 4 en sigmas

Ademas guarda unos frames sueltos de muestra (para el preview del paso 7, que
compara "acumulado bonito" vs "frame individual").

    uv run python debug/pre_pasada.py [--force]
"""
import argparse
import time

import cv2
import numpy as np

from pre_common import (CLIP, START, CACHE, SIGMA_FLOOR, ensure_dirs,
                        estimate_motion, warp_to, illum_stats, match_illum)
import pre_fondo

CACHE_FILE = CACHE / "pasada.npz"
SAMPLE_FRAMES = [60, 80, 100, 140, 200, 300]     # frames sueltos de muestra


def run(clip=CLIP, start=START):
    fondo = pre_fondo.load()
    bg_mean = fondo["bg_mean"].astype(np.float32)
    bg_sigma = np.maximum(fondo["bg_sigma"], SIGMA_FLOOR).astype(np.float32)
    diff_sigma = np.maximum(fondo["diff_sigma"], SIGMA_FLOOR).astype(np.float32)
    ref_med = float(fondo["ref_med"])
    ref_mad = float(fondo["ref_mad"])

    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shape = (h, w)

    bg_u8 = np.clip(bg_mean, 0, 255).astype(np.uint8)

    acc_ff = np.zeros(shape, np.float32)
    acc_zff = np.zeros(shape, np.float32)
    acc_bg = np.zeros(shape, np.float32)
    acc_zbg = np.zeros(shape, np.float32)
    cover = np.zeros(shape, np.float32)      # cuantas veces el pixel supero 4 sigmas

    ones = np.ones(shape, np.float32)
    samples = {}
    prev_al = None
    M_prev = None
    n_fallback = 0
    idx = 0
    t0 = time.time()

    print(f"Pasada sobre {nfr} frames ({w}x{h}); detonacion en {start}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             (3, 3), 0).astype(np.float32)

        # registro DIRECTO contra el fondo -> sin deriva acumulada
        M = estimate_motion(g, bg_u8)
        if M is None:
            M = M_prev                    # el humo tapo demasiado: reusa el anterior
            n_fallback += 1
        else:
            M_prev = M

        g_al = warp_to(g, M, shape)
        valid = warp_to(ones, M, shape) > 0.99
        g_al = match_illum(g_al, ref_med, ref_mad, mask=valid)
        g_al[~valid] = bg_mean[~valid]     # borde del warp: neutro (diff = 0)

        d_bg = np.abs(g_al - bg_mean)
        z_bg = d_bg / bg_sigma

        if prev_al is not None:
            d_ff = np.abs(g_al - prev_al)
            z_ff = d_ff / diff_sigma
        else:
            d_ff = np.zeros(shape, np.float32)
            z_ff = np.zeros(shape, np.float32)

        if idx >= start:
            np.maximum(acc_ff, d_ff, out=acc_ff)
            np.maximum(acc_zff, z_ff, out=acc_zff)
            np.maximum(acc_bg, d_bg, out=acc_bg)
            np.maximum(acc_zbg, z_bg, out=acc_zbg)
            cover += (z_ff > 4.0)

        if idx in SAMPLE_FRAMES:
            samples[f"ff_{idx}"] = np.clip(d_ff, 0, 255).astype(np.uint8)
            samples[f"zff_{idx}"] = np.clip(z_ff * 4, 0, 255).astype(np.uint8)
            samples[f"bg_{idx}"] = np.clip(d_bg, 0, 255).astype(np.uint8)

        prev_al = g_al
        idx += 1
        if idx % 50 == 0:
            print(f"  {idx}/{nfr}  ({time.time() - t0:.0f}s)")

    cap.release()
    print(f"  listo en {time.time() - t0:.0f}s; "
          f"frames sin registro propio: {n_fallback}")

    np.savez_compressed(
        CACHE_FILE, acc_ff=acc_ff, acc_zff=acc_zff, acc_bg=acc_bg,
        acc_zbg=acc_zbg, cover=cover.astype(np.float32),
        nframes=idx, sample_ids=np.array(SAMPLE_FRAMES), **samples,
    )
    print(f"  cache -> {CACHE_FILE.name}")


def load():
    z = np.load(CACHE_FILE)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    if CACHE_FILE.exists() and not args.force:
        print(f"Ya existe {CACHE_FILE.name} (--force para recomputar)")
    else:
        run()
