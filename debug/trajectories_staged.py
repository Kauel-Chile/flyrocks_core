"""
Trayectorias por ETAPAS. En vez de filtrar por VELOCIDAD (que mata las parabolas
lentas), filtra por SUAVIDAD: que tan bien la traza se ajusta a una curva suave
(recta o parabola). Roca = ajusta bien (residuo bajo); polvo = erratico (residuo
alto). NO penaliza parabolas. El AZUL pintado nunca se cae (garantia).

Salidas en debug/out/6_mascaras/trayectorias/:
  1_dedup.png       -> todas, deduplicadas por sector (recall alto, con parabolas)
  2_suaves.png      -> solo las que ajustan a curva suave (roca) + azules
  3_extendida.png   -> ajustadas + extendidas al tiro (sobre el pozo)

    uv run python debug/trajectories_staged.py [--max-res 18]
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from clean_and_stitch import (load_tracks, dedup, fit_and_extend,   # noqa: E402
                              render, render_fit, load_shots)
from paint_filter import PaintMask                                  # noqa: E402

OUT = ROOT / "debug" / "out" / "6_mascaras" / "trayectorias"
GRAY = ROOT / "debug" / "out" / "6_mascaras" / "1_intensidad.png"
REAL = ROOT / "debug" / "out" / "4_fase0_referencias" / "frame_full_res_13s.png"
PAINT = ROOT / "debug" / "out" / "4_fase0_referencias" / "lienzo_para_pintar_gris - Copy.png"


def fit_residual(arr):
    """Residuo (px) de ajustar la traza a una curva suave (recta/parabola) en su
    eje principal. Bajo = suave (roca); alto = erratico (polvo)."""
    P = arr[:, :2].astype(float)
    if len(P) < 4:
        return 0.0                                  # muy corta: no se juzga
    c = P.mean(0)
    d = P - c
    w, V = np.linalg.eigh(d.T @ d)
    u = V[:, int(np.argmax(w))]
    t = d @ u
    v = np.array([-u[1], u[0]])
    s = d @ v
    deg = 2 if len(P) >= 5 else 1
    poly = np.poly1d(np.polyfit(t, s, deg))
    return float(np.sqrt(np.mean((s - poly(t)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-res", type=float, default=18.0,
                    help="residuo maximo (px) para considerar la traza suave (roca)")
    cfg = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tracks, origin = load_tracks()
    bg = cv2.imread(str(GRAY))
    if bg is None:
        bg = np.zeros((2160, 3840, 3), np.uint8)
    real = cv2.imread(str(REAL))
    shots_px, shot_frames = load_shots()
    pm = PaintMask(PAINT) if PAINT.exists() else None

    # etapa 1: dedup de TODAS (recall alto, conserva parabolas)
    base = dedup(tracks, origin, 0.03)
    render(base, origin, OUT / "1_dedup.png", bg)
    print(f"[1] dedup todas: {len(tracks)} -> {len(base)}")

    # etapa 2: suavidad (ajuste a curva) — NO por velocidad
    res = np.array([fit_residual(a) for a in base])
    print(f"Residuo de ajuste px: p25={np.percentile(res,25):.1f} "
          f"mediana={np.median(res):.1f} p75={np.percentile(res,75):.1f} "
          f"p90={np.percentile(res,90):.1f}")

    def is_blue(a):
        return pm is not None and pm.in_blue(a[:, 0], a[:, 1]).any()

    smooth = [a for a, r in zip(base, res) if r <= cfg.max_res or is_blue(a)]
    render(smooth, origin, OUT / "2_suaves.png", bg)
    print(f"[2] suaves (res<= {cfg.max_res} o azul): {len(base)} -> {len(smooth)}")

    # etapa 3: parabola + extension al tiro (sobre el pozo)
    e = fit_and_extend(smooth, shots_px, shot_frames, origin)
    render_fit(e, shots_px, OUT / "3_extendida.png", real if real is not None else bg)
    n_ext = sum(1 for _, ex, _ in e if ex is not None)
    print(f"[3] extendidas al tiro: {n_ext}/{len(e)}  -> {OUT}")


if __name__ == "__main__":
    main()
