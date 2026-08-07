"""
Camino 1: extrae trayectorias DIRECTO del campo de estelas (sin tracking
frame-a-frame). Aprovecha que las estelas son radiales:

  1. Toma los pixeles de estela (gris > umbral) AFUERA del area de polvo.
  2. Los pasa a polares (r, theta) alrededor del origen.
  3. Agrupa por corredor angular fino; dentro de cada angulo, separa rocas por
     saltos grandes en r. Cada segmento = una trayectoria radial limpia.

Cachea los campos (gray, persistencia, origen) para iterar rapido.

    uv run python debug/polar_extract.py --clip <clip_4k.mp4>
        [--nbins 2000 --gap 140 --min-ext 70 --thresh 16]
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "debug"))
from harness import extract_events, CLIP_PATH, OUT_DIR                # noqa: E402

C1_DIR = OUT_DIR / "1_MASCARA_camino1"
CACHE_DIR = OUT_DIR / "_cache"
C1_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_fields(clip, force=False):
    cache = CACHE_DIR / "fields.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        return d["gray"], d["pers"], d["origin"], tuple(d["meta"])
    tensor, meta = extract_events(Path(clip), stabilize=True)
    w, h = int(meta[0]), int(meta[1])
    xs = np.clip(tensor[:, 0].astype(int), 0, w - 1)
    ys = np.clip(tensor[:, 1].astype(int), 0, h - 1)
    gray = np.zeros((h, w), np.float32)
    np.maximum.at(gray, (ys, xs), tensor[:, 3])
    pers = np.zeros((h, w), np.int32)
    np.add.at(pers, (ys, xs), 1)
    yy, xx = np.nonzero(pers)
    wgt = pers[yy, xx]
    origin = np.array([np.average(xx, weights=wgt), np.average(yy, weights=wgt)])
    np.savez_compressed(cache, gray=gray, pers=pers, origin=origin,
                        meta=np.array(meta))
    return gray, pers, origin, tuple(meta)


def extract_trajectories(gray, pers, origin, cfg):
    h, w = gray.shape
    blast = (pers >= cfg.persist).astype(np.uint8) * 255
    k = cfg.dilate
    blast = cv2.dilate(blast, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    ys, xs = np.nonzero((gray > cfg.thresh) & (blast == 0))
    dx = xs - origin[0]
    dy = ys - origin[1]
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    bin_id = ((th + np.pi) / (2 * np.pi) * cfg.nbins).astype(int) % cfg.nbins

    trajs = []
    order = np.argsort(bin_id, kind="stable")
    bin_s = bin_id[order]
    xs_s, ys_s, r_s = xs[order], ys[order], r[order]
    edges = np.searchsorted(bin_s, np.arange(cfg.nbins + 1))

    for b in range(cfg.nbins):
        lo, hi = edges[b], edges[b + 1]
        if hi - lo < 4:
            continue
        rr = r_s[lo:hi]
        xi, yi = xs_s[lo:hi], ys_s[lo:hi]
        o = np.argsort(rr)
        rr, xi, yi = rr[o], xi[o], yi[o]
        # separar rocas distintas por saltos grandes en radio
        splits = np.where(np.diff(rr) > cfg.gap)[0] + 1
        for seg_idx in np.split(np.arange(len(rr)), splits):
            if len(seg_idx) < 4:
                continue
            rseg = rr[seg_idx]
            if rseg[-1] - rseg[0] < cfg.min_ext:
                continue
            # centerline: mediana de posicion en ~12 tramos de radio
            xseg, yseg = xi[seg_idx], yi[seg_idx]
            nb = min(12, max(2, len(seg_idx) // 3))
            qs = np.linspace(rseg[0], rseg[-1], nb + 1)
            pts = []
            for a, bnd in zip(qs[:-1], qs[1:]):
                m = (rseg >= a) & (rseg <= bnd)
                if m.any():
                    pts.append((np.median(xseg[m]), np.median(yseg[m])))
            if len(pts) >= 2:
                trajs.append(np.array(pts))
    return trajs, blast


def color_for(i):
    c = cv2.applyColorMap(np.uint8([[(i * 41) % 256]]), cv2.COLORMAP_HSV)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=str, default=str(CLIP_PATH))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--persist", type=int, default=6)
    ap.add_argument("--dilate", type=int, default=30)
    ap.add_argument("--thresh", type=float, default=16.0)
    ap.add_argument("--nbins", type=int, default=2000)
    ap.add_argument("--gap", type=float, default=140.0)
    ap.add_argument("--min-ext", type=float, default=70.0)
    cfg = ap.parse_args()

    gray, pers, origin, meta = get_fields(cfg.clip, cfg.force)
    w, h = int(meta[0]), int(meta[1])
    trajs, blast = extract_trajectories(gray, pers, origin, cfg)
    print(f"[camino1] {len(trajs)} trayectorias extraidas del campo de estelas")

    gray_bg = cv2.cvtColor(
        (np.clip(gray / (gray.max() + 1e-6), 0, 1) ** 0.5 * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(C1_DIR / "mascara_gris_estelas.png"), gray_bg)
    thick = 3 if w > 2500 else 2
    over = gray_bg.copy()
    for i, a in enumerate(trajs):
        poly = a.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(over, [poly], False, color_for(i), thick, cv2.LINE_AA)
    cont, _ = cv2.findContours(blast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cont, -1, (0, 0, 255), 1)
    cv2.imwrite(str(C1_DIR / "trayectorias_sobre_gris.png"), over)

    black = np.zeros((h, w, 3), np.uint8)
    for i, a in enumerate(trajs):
        poly = a.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(black, [poly], False, color_for(i), thick, cv2.LINE_AA)
    cv2.imwrite(str(C1_DIR / "trayectorias.png"), black)
    print(f"[viz] -> {C1_DIR}")


if __name__ == "__main__":
    main()
