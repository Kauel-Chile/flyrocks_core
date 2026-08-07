"""
Deduplicacion/costura de trayectorias (sobre tracks cacheados, sin re-extraer).

La ventana larga trackea la misma estela varias veces (roca + puffs de polvo que
la siguen). Duplicados = lineas casi identicas: mismo angulo de vuelo y misma
distancia perpendicular al origen. Se agrupan y se deja una representante (la mas
larga) por estela. Tambien fusiona fragmentos de la misma linea.

    uv run python debug/dedup_view.py [--da 0.02 --dd 12]
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "debug" / "out" / "_cache"
GRAY = ROOT / "debug" / "out" / "0_diagnostico" / "comparacion_4k" / "5_MI_gris_igualada_g085.png"
OUT = ROOT / "debug" / "out" / "2_comparacion_camino2"


def line_signature(arr, origin):
    p0, p1 = arr[0, :2], arr[-1, :2]
    d = p1 - p0
    n = np.hypot(*d)
    if n < 1e-6:
        return None
    ux, uy = d / n
    a = np.arctan2(uy, ux)                       # angulo de vuelo (dirigido)
    v = origin - p0
    perp = abs(v[0] * uy - v[1] * ux)            # dist perpendicular al origen
    length = n
    return a, perp, length


def dedup(tracks, origin, da, dd):
    groups = {}
    for arr in tracks:
        sig = line_signature(arr, origin)
        if sig is None:
            continue
        a, perp, length = sig
        key = (round(a / da), round(perp / dd))
        if key not in groups or length > groups[key][1]:
            groups[key] = (arr, length)
    return [g[0] for g in groups.values()]


def render(tracks, origin, out_png, bg=None):
    if bg is not None:
        canvas = bg.copy()
    else:
        canvas = np.zeros((2160, 3840, 3), np.uint8)
    h, w = canvas.shape[:2]
    for i, arr in enumerate(tracks):
        c = cv2.applyColorMap(np.uint8([[(i * 41) % 256]]), cv2.COLORMAP_HSV)[0, 0]
        poly = arr[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, (int(c[0]), int(c[1]), int(c[2])),
                      3, cv2.LINE_AA)
    cv2.circle(canvas, tuple(origin.astype(int)), 12, (255, 255, 255), 2)
    cv2.imwrite(str(out_png), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da", type=float, default=0.02, help="tolerancia angular (rad)")
    ap.add_argument("--dd", type=float, default=12.0, help="tolerancia lateral (px)")
    cfg = ap.parse_args()

    d = np.load(CACHE / "polar_result.npz")
    traj, origin = d["traj"], d["origin"]
    tracks = [traj[traj[:, 0] == i][:, 1:4] for i in np.unique(traj[:, 0])]
    tracks = [t[np.argsort(t[:, 2])] for t in tracks]

    kept = dedup(tracks, origin, cfg.da, cfg.dd)
    print(f"Dedup: {len(tracks)} -> {len(kept)} trayectorias (da={cfg.da}, dd={cfg.dd})")

    bg = cv2.imread(str(GRAY)) if GRAY.exists() else None
    render(kept, origin, OUT / "trayectorias_dedup_sobre_gris.png", bg)
    render(kept, origin, OUT / "trayectorias_dedup.png", None)
    print(f"[viz] -> {OUT}/trayectorias_dedup_sobre_gris.png")


if __name__ == "__main__":
    main()
