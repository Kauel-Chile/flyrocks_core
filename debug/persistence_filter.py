"""
Filtro por PERSISTENCIA (roca transitoria vs polvo persistente).

Una roca -aunque sea lenta/parabolica- pasa por cada pixel POCOS frames
(transitoria). El polvo se QUEDA (persiste). No depende de la velocidad -> no
mata parabolas. Filtra las trazas por la persistencia media de sus pixeles.

Salidas en debug/out/6_mascaras/trayectorias/:
  2b_persistencia.png          -> dedup filtrado por persistencia (roca)
  3b_extendida_persistencia.png-> + parabola + extension al tiro (sobre el pozo)

    uv run python debug/persistence_filter.py [--max-pers 12]
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from harness import estimate_global_motion, NOISE_THRESHOLD          # noqa: E402
from clean_and_stitch import (load_tracks, dedup, fit_and_extend,     # noqa: E402
                              render, render_fit, load_shots)
from paint_filter import PaintMask                                    # noqa: E402

SCRATCH = Path(
    r"C:\Users\carlo\AppData\Local\Temp\claude"
    r"\D--PROYECTOS-Enaex---Flyrocks-detovision-standalone-flyrocks-core"
    r"\3f0c6b88-01e1-4a43-9f40-09651754d37b\scratchpad"
)
CLIP = SCRATCH / "clip_full.mp4"
START = 48
CACHE = ROOT / "debug" / "out" / "_cache"
OUT = ROOT / "debug" / "out" / "6_mascaras" / "trayectorias"
GRAY = ROOT / "debug" / "out" / "6_mascaras" / "1_intensidad.png"
REAL = ROOT / "debug" / "out" / "4_fase0_referencias" / "frame_full_res_13s.png"
PAINT = ROOT / "debug" / "out" / "4_fase0_referencias" / "lienzo_para_pintar_gris - Copy.png"


def build_persistence(clip, force=False):
    cache = CACHE / "persistence.npy"
    if cache.exists() and not force:
        print("[cache] persistence.npy")
        return np.load(cache)
    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, prev = cap.read()
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    pers = np.zeros((h, w), np.int32)
    idx = 2
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        cg = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        M = estimate_global_motion(prev_gray, cg)
        ref = cv2.warpAffine(prev_gray, M, (w, h)) if M is not None else prev_gray
        diff = cv2.absdiff(ref, cg)
        if idx >= START:
            pers += (diff > NOISE_THRESHOLD).astype(np.int32)
        prev_gray = cg
        idx += 1
    cap.release()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, pers)
    print(f"[persistencia] calculada y cacheada ({idx-START} frames)")
    return pers


def track_persistence(arr, pers):
    px = np.clip(arr[:, 0].astype(int), 0, pers.shape[1] - 1)
    py = np.clip(arr[:, 1].astype(int), 0, pers.shape[0] - 1)
    return float(np.median(pers[py, px]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pers", type=float, default=12.0,
                    help="persistencia media maxima para ser roca (frames)")
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()

    pers = build_persistence(CLIP, cfg.force)
    tracks, origin = load_tracks()
    bg = cv2.imread(str(GRAY))
    if bg is None:
        bg = np.zeros((2160, 3840, 3), np.uint8)
    real = cv2.imread(str(REAL))
    shots_px, shot_frames = load_shots()
    pm = PaintMask(PAINT) if PAINT.exists() else None

    base = dedup(tracks, origin, 0.03)
    pv = np.array([track_persistence(a, pers) for a in base])
    print(f"Persistencia media por traza (frames): p25={np.percentile(pv,25):.0f} "
          f"mediana={np.median(pv):.0f} p75={np.percentile(pv,75):.0f} "
          f"p90={np.percentile(pv,90):.0f}")

    def is_blue(a):
        return pm is not None and pm.in_blue(a[:, 0], a[:, 1]).any()

    keep = [a for a, p in zip(base, pv) if p <= cfg.max_pers or is_blue(a)]
    render(keep, origin, OUT / "2b_persistencia.png", bg)
    print(f"[persistencia] dedup {len(base)} -> {len(keep)} (roca; <= {cfg.max_pers})")

    e = fit_and_extend(keep, shots_px, shot_frames, origin)
    render_fit(e, shots_px, OUT / "3b_extendida_persistencia.png",
               real if real is not None else bg)
    n_ext = sum(1 for _, ex, _ in e if ex is not None)
    print(f"[persistencia] extendidas: {n_ext}/{len(e)}  -> {OUT}")


if __name__ == "__main__":
    main()
