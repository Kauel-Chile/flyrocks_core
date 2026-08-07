"""
Filtro de BORDE DE ATAQUE usando la mascara temporal (primera llegada).

Una roca es lo PRIMERO que toca sus pixeles; el polvo que la sigue llega DESPUES.
Por traza: lag = mediana( t_de_la_traza - primera_llegada[pixel] ).
  lag ~ 0  -> la traza fue el borde de ataque = ROCA (aunque sea lenta/parabola)
  lag alto -> algo paso antes = polvo/estela seguidora
No depende de la velocidad -> NO mata parabolas.

Pipeline: todas -> filtro leading-edge -> dedup inteligente. Deja imagen para ver.
    uv run python debug/exp_temporal_leading.py [--max-lag 12]
"""
import sys
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from harness import estimate_global_motion, NOISE_THRESHOLD          # noqa: E402
from clean_and_stitch import load_tracks, render                     # noqa: E402
from paint_filter import PaintMask                                   # noqa: E402

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
PAINT = ROOT / "debug" / "out" / "4_fase0_referencias" / "lienzo_para_pintar_gris - Copy.png"


def build_first(clip, force=False):
    cache = CACHE / "first_arrival.npy"
    if cache.exists() and not force:
        return np.load(cache)
    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, prev = cap.read()
    pg = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    first = np.full((h, w), -1, np.int32); idx = 2
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        cg = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        M = estimate_global_motion(pg, cg)
        ref = cv2.warpAffine(pg, M, (w, h)) if M is not None else pg
        diff = cv2.absdiff(ref, cg)
        if idx >= START:
            newly = (diff > NOISE_THRESHOLD) & (first < 0)
            first[newly] = idx
        pg = cg; idx += 1
    cap.release()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, first)
    return first


def sig(a, o):
    d = a[-1, :2] - a[0, :2]
    n = np.hypot(*d)
    if n < 1:
        return None
    u = d / n
    perp = (o[0] - a[0, 0]) * u[1] - (o[1] - a[0, 1]) * u[0]
    r = np.hypot(a[:, 0] - o[0], a[:, 1] - o[1])
    return np.arctan2(u[1], u[0]), perp, r.min(), r.max()


def smart_dedup(tracks, origin, dtheta=0.03, dperp=30):
    S = [sig(a, origin) for a in tracks]
    par = list(range(len(tracks)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    buckets = defaultdict(list)
    for i, s in enumerate(S):
        if s is not None:
            buckets[round(s[0] / 0.05)].append(i)
    for key, idxs in buckets.items():
        cand = idxs + buckets.get(key + 1, []) + buckets.get(key - 1, [])
        for i in idxs:
            for j in cand:
                if j <= i:
                    continue
                si, sj = S[i], S[j]
                dang = abs(np.arctan2(np.sin(si[0] - sj[0]), np.cos(si[0] - sj[0])))
                if dang < dtheta and abs(si[1] - sj[1]) < dperp \
                        and max(si[2], sj[2]) < min(si[3], sj[3]):
                    par[find(i)] = find(j)
    cl = defaultdict(list)
    for i in range(len(tracks)):
        if S[i] is not None:
            cl[find(i)].append(i)
    ext = lambda a: np.hypot(a[-1, 0] - a[0, 0], a[-1, 1] - a[0, 1])
    return [max((tracks[i] for i in c), key=ext) for c in cl.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lag", type=float, default=12.0)
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()

    first = build_first(CLIP, cfg.force)
    tracks, origin = load_tracks()
    pm = PaintMask(PAINT) if PAINT.exists() else None

    def lag(a):
        px = np.clip(a[:, 0].astype(int), 0, first.shape[1] - 1)
        py = np.clip(a[:, 1].astype(int), 0, first.shape[0] - 1)
        f = first[py, px]
        v = f >= 0
        return float(np.median(a[v, 2] - f[v])) if v.sum() >= 2 else 999

    lags = np.array([lag(a) for a in tracks])
    print(f"Lag (t - primera_llegada) frames: p25={np.percentile(lags,25):.0f} "
          f"mediana={np.median(lags):.0f} p75={np.percentile(lags,75):.0f}")

    def is_blue(a):
        return pm is not None and pm.in_blue(a[:, 0], a[:, 1]).any()

    lead = [a for a, l in zip(tracks, lags) if l <= cfg.max_lag or is_blue(a)]
    print(f"[leading-edge] {len(tracks)} -> {len(lead)} (roca; lag<= {cfg.max_lag})")
    kept = smart_dedup(lead, origin)
    print(f"[dedup smart] -> {len(kept)}")

    bg = cv2.imread(str(GRAY))
    render(kept, origin, OUT / "exp_leading_edge.png", bg)
    print(f"-> {OUT / 'exp_leading_edge.png'}")


if __name__ == "__main__":
    main()
