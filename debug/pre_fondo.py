"""
PASO 2 — Modelo de fondo del pre-roll ("pantalla verde").

Usa los frames ANTERIORES a la detonacion (0..START-1) para caracterizar la
escena estatica y, sobre todo, EL RUIDO de cada pixel.

Produce (cacheado en 7_preproceso/_cache/fondo.npz):
  bg_mean    -> imagen media del terreno sin humo (el "fondo limpio")
  bg_sigma   -> desviacion estandar POR PIXEL del valor (ruido de reposo)
  diff_sigma -> desviacion estandar POR PIXEL del absdiff frame-a-frame
                (= el ruido real del proceso de deteccion actual)
  valid      -> pixeles con datos en todos los frames (sin borde de warp)

Imagenes en 7_preproceso/01_fondo/:
  fondo_media.png             el fondo limpio
  fondo_sigma.png             mapa de ruido en falso color
  fondo_umbral_equivalente.png  A CUANTOS SIGMAS equivale el umbral fijo de 8
                              en cada pixel  <-- el diagnostico importante

    uv run python debug/pre_fondo.py [--force]
"""
import argparse

import cv2
import numpy as np

from pre_common import (CLIP, START, CACHE, D_FONDO, NOISE_THRESHOLD,
                        SIGMA_FLOOR, ensure_dirs, estimate_motion, warp_to,
                        compose, invert, illum_stats, match_illum,
                        to8, falsecolor, label, side_by_side, save)

CACHE_FILE = CACHE / "fondo.npz"


def build(clip=CLIP, start=START):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir {clip}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shape = (h, w)

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Clip vacio")
    ref = cv2.GaussianBlur(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
                           (3, 3), 0).astype(np.float32)
    ref_med, ref_mad = illum_stats(ref)

    # acumuladores (float64: 48 frames en 4K, la precision importa para sigma)
    n = np.zeros(shape, np.float64)
    s1 = np.zeros(shape, np.float64)
    s2 = np.zeros(shape, np.float64)
    # ruido del absdiff frame-a-frame (el que usa el pipeline hoy)
    dn = np.zeros(shape, np.float64)
    d1 = np.zeros(shape, np.float64)
    d2 = np.zeros(shape, np.float64)

    ones = np.ones(shape, np.float32)
    s1 += ref
    s2 += ref.astype(np.float64) ** 2
    n += 1.0

    M_acc = np.array([[1, 0, 0], [0, 1, 0]], np.float32)   # frame_i -> ref
    prev_gray = ref
    prev_aligned = ref
    idx = 1
    print(f"Pre-roll: frames 0..{start - 1}  ({w}x{h})")

    while idx < start:
        ret, frame = cap.read()
        if not ret:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             (3, 3), 0).astype(np.float32)

        # movimiento prev->curr; queremos curr->ref = (prev->ref) o (curr->prev)
        A = estimate_motion(prev_gray, g)
        if A is not None:
            M_acc = compose(M_acc, invert(A))

        g_al = warp_to(g, M_acc, shape)
        valid = warp_to(ones, M_acc, shape) > 0.99
        g_al = match_illum(g_al, ref_med, ref_mad, mask=valid)

        s1[valid] += g_al[valid]
        s2[valid] += g_al[valid].astype(np.float64) ** 2
        n[valid] += 1.0

        d = np.abs(g_al - prev_aligned)
        vd = valid
        d1[vd] += d[vd]
        d2[vd] += d[vd].astype(np.float64) ** 2
        dn[vd] += 1.0

        prev_gray = g
        prev_aligned = g_al
        idx += 1

    cap.release()

    nn = np.maximum(n, 1.0)
    bg_mean = (s1 / nn).astype(np.float32)
    var = np.maximum(s2 / nn - (s1 / nn) ** 2, 0.0)
    bg_sigma = np.sqrt(var).astype(np.float32)

    dnn = np.maximum(dn, 1.0)
    dvar = np.maximum(d2 / dnn - (d1 / dnn) ** 2, 0.0)
    diff_sigma = np.sqrt(dvar).astype(np.float32)
    diff_mean = (d1 / dnn).astype(np.float32)

    valid_all = (n >= idx * 0.9)

    print(f"  frames usados: {idx}")
    print(f"  bg_sigma   : mediana={np.median(bg_sigma[valid_all]):.2f} "
          f"p95={np.percentile(bg_sigma[valid_all], 95):.2f}")
    print(f"  diff_sigma : mediana={np.median(diff_sigma[valid_all]):.2f} "
          f"p95={np.percentile(diff_sigma[valid_all], 95):.2f}")

    np.savez_compressed(
        CACHE_FILE, bg_mean=bg_mean, bg_sigma=bg_sigma,
        diff_sigma=diff_sigma, diff_mean=diff_mean,
        valid=valid_all, ref_med=ref_med, ref_mad=ref_mad, nframes=idx,
    )
    print(f"  cache -> {CACHE_FILE.name}")
    return dict(bg_mean=bg_mean, bg_sigma=bg_sigma, diff_sigma=diff_sigma,
                diff_mean=diff_mean, valid=valid_all)


def load():
    z = np.load(CACHE_FILE)
    return {k: z[k] for k in z.files}


def render(d):
    bg_mean, bg_sigma = d["bg_mean"], d["bg_sigma"]
    diff_sigma, valid = d["diff_sigma"], d["valid"]

    save(D_FONDO / "fondo_media.png", to8(bg_mean, 0.5, 99.5))

    # --- mapa de ruido -------------------------------------------------
    hi = float(np.percentile(bg_sigma[valid], 99.0))
    sig8 = to8(bg_sigma, lo=0.0, hi=max(hi, 1.0))
    save(D_FONDO / "fondo_sigma.png",
         label(falsecolor(sig8),
               f"sigma por pixel (azul=quieto, rojo=ruidoso)  0..{hi:.1f} niveles"))

    dhi = float(np.percentile(diff_sigma[valid], 99.0))
    dsig8 = to8(diff_sigma, lo=0.0, hi=max(dhi, 1.0))
    save(D_FONDO / "fondo_sigma_diff.png",
         label(falsecolor(dsig8),
               f"sigma del absdiff frame-a-frame  0..{dhi:.1f}"))

    # --- EL diagnostico: a cuantos sigmas equivale el umbral fijo de 8 ---
    eq = NOISE_THRESHOLD / np.maximum(diff_sigma, SIGMA_FLOOR)
    top = float(np.percentile(eq[valid], 98))
    eq8 = to8(np.clip(eq, 0, top), lo=0.0, hi=top)
    vis = falsecolor(eq8, cv2.COLORMAP_TURBO)
    vis = label(vis, f"umbral fijo {NOISE_THRESHOLD} expresado en SIGMAS locales "
                     f"(escala 0..{top:.0f})")
    vis = label(vis, "azul = pocos sigmas ahi (umbral permisivo)", org=(30, 150))
    vis = label(vis, "rojo = MUCHOS sigmas (ciego: se pierden rocas tenues)",
                org=(30, 230))
    save(D_FONDO / "fondo_umbral_equivalente.png", vis)

    v = eq[valid]
    print("\n  El umbral fijo de 8, medido en sigmas locales:")
    for p in (5, 25, 50, 75, 95):
        print(f"    p{p:<3} = {np.percentile(v, p):6.1f} sigmas")
    frac_ciego = float((v > 6).mean()) * 100
    frac_ruido = float((v < 2).mean()) * 100
    print(f"    -> {frac_ciego:.1f}% del cuadro con umbral >6 sigmas (ciego a lo tenue)")
    print(f"    -> {frac_ruido:.1f}% del cuadro con umbral <2 sigmas (deja pasar ruido)")
    for k in (2.0, 3.0, 4.0, 5.0):
        equiv = np.percentile(k * np.maximum(diff_sigma[valid], SIGMA_FLOOR), 50)
        print(f"    un umbral de {k:.0f} sigmas equivaldria a ~{equiv:.1f} "
              f"niveles de intensidad (hoy: {NOISE_THRESHOLD})")

    # OJO: side_by_side vuelve a etiquetar, asi que aqui se pasan imagenes crudas
    save(D_FONDO / "fondo_resumen.png", side_by_side([
        (cv2.cvtColor(to8(bg_mean, 0.5, 99.5), cv2.COLOR_GRAY2BGR),
         "fondo medio (pre-roll)"),
        (falsecolor(dsig8), "ruido del diff: azul=quieto, rojo=ruidoso"),
        (falsecolor(eq8), "umbral 8 en sigmas: rojo = ciego a lo tenue"),
    ]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    if CACHE_FILE.exists() and not args.force:
        print(f"Usando cache {CACHE_FILE.name} (--force para recomputar)")
        d = load()
    else:
        d = build()
    render(d)
