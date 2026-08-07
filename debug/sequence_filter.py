"""
Filtro espacio-temporal por secuencia de tiros (sobre datos cacheados).

Usa:
  out/_cache/polar_result.npz  -> tracks del Camino 2 [id,x,y,t]
  out/_cache/fields.npz        -> gray, persistencia (area polvo), meta
  debug/Secuencia (2).csv      -> tiros (X,Y mina, DetonatingTime ms)

1. Alinea el patron de tiros al footprint de polvo (PCA + escala), y desambigua
   la orientacion con las pistas del cliente: la detonacion parte abajo-derecha
   (~13.5s = frame ~30) y un tiro muy a la izquierda detona ~15s (frame ~75).
2. A cada tiro le asigna su frame de detonacion.
3. Filtra: una roca es valida si su trayectoria, extendida hacia atras, apunta a
   un tiro que YA habia detonado cuando la roca nace.

    uv run python debug/sequence_filter.py [--first-frame 30 --scale 0.6]
"""
import sys
import csv
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "debug" / "out"
CACHE = OUT_DIR / "_cache"
SEQ_DIR = OUT_DIR / "3_secuencia_filtro"
SEQ_DIR.mkdir(parents=True, exist_ok=True)
FPS = 30.0


def load_tracks():
    d = np.load(CACHE / "polar_result.npz")
    traj = d["traj"]
    tracks = []
    for i in np.unique(traj[:, 0]):
        a = traj[traj[:, 0] == i][:, 1:4].astype(float)
        tracks.append(a[np.argsort(a[:, 2])])
    return tracks


def load_holes():
    rows = list(csv.reader(open(ROOT / "debug" / "Secuencia (2).csv")))
    X, Y, T = [], [], []
    for r in rows[1:]:
        if len(r) >= 5 and r[4].strip().replace('.', '').isdigit():
            X.append(float(r[1])); Y.append(float(r[2])); T.append(float(r[4]))
    return np.column_stack([X, Y]), np.array(T)


def pca(pts):
    m = pts.mean(0)
    cov = np.cov((pts - m).T)
    val, vec = np.linalg.eigh(cov)
    order = val.argsort()[::-1]
    return m, vec[:, order], val[order]


def apply_H(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def align(P, T, dust, scale):
    ys, xs = np.nonzero(dust)
    D = np.column_stack([xs, ys]).astype(float)
    mP, EP, lP = pca(P)
    mD, ED, lD = pca(D)
    s = np.sqrt(lD[0] / lP[0]) * scale

    i_early = np.argmin(T)                     # primer tiro (abajo-derecha)
    i_left = np.argmin(np.abs(T - 4500))       # tiro que detona ~15s (izquierda)
    best, best_score = None, -1e18
    for sx in (1, -1):
        for sy in (1, -1):
            R = ED @ np.diag([sx, sy]).astype(float) @ EP.T
            Pp = mD + s * ((P - mP) @ R.T)
            # primer tiro abajo-derecha (x grande, y grande) ; ~15s a la izq (x chico)
            score = Pp[i_early, 0] + Pp[i_early, 1] - 1.5 * Pp[i_left, 0]
            if score > best_score:
                best_score, best = score, Pp
    return best, mD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-frame", type=float, default=48.0,
                    help="frame del primer tiro en el clip (usuario: t14.10s = f48)")
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--persist", type=int, default=6)
    ap.add_argument("--dilate", type=int, default=30)
    ap.add_argument("--hole-dist", type=float, default=220.0,
                    help="max distancia perpendicular traza->tiro (px)")
    ap.add_argument("--max-back", type=float, default=1600.0)
    ap.add_argument("--margin", type=float, default=12.0,
                    help="margen de frames (vuelo tiro->borde de polvo)")
    cfg = ap.parse_args()

    f = np.load(CACHE / "fields.npz")
    gray, pers, meta = f["gray"], f["pers"], f["meta"]
    w, h = int(meta[0]), int(meta[1])
    dust = (pers >= cfg.persist).astype(np.uint8) * 255
    dust = cv2.dilate(dust, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.dilate, cfg.dilate)))

    tracks = load_tracks()
    P, T = load_holes()
    # homografia PRECISA (correspondencias del usuario) si existe; si no, PCA aprox
    hpath = CACHE / "homografia.npz"
    if hpath.exists():
        H = np.load(hpath)["H"]
        Hpx = apply_H(H, P)
        print("[H] usando homografia precisa (_cache/homografia.npz)")
    else:
        Hpx, _ = align(P, T, dust, cfg.scale)
        print("[H] usando alineacion PCA aproximada (sin homografia)")
    hole_frame = cfg.first_frame + (T - T.min()) / 1000 * FPS
    print(f"{len(tracks)} tracks | {len(P)} tiros | detonan frame "
          f"{hole_frame.min():.0f}..{hole_frame.max():.0f}")

    keep, drop = [], []
    for a in tracks:
        p0 = a[0, :2]
        t0 = a[0, 2]
        k = min(3, len(a) - 1)
        d = a[k, :2] - p0
        n = np.hypot(*d)
        if n < 1e-6:
            drop.append(a); continue
        d /= n
        rel = Hpx - p0
        tproj = rel @ (-d)                         # cuanto atras de p0
        perp = np.linalg.norm(rel - np.outer(tproj, -d), axis=1)
        behind = (tproj > 0) & (tproj < cfg.max_back)
        if not behind.any():
            drop.append(a); continue
        cand = np.where(behind)[0]
        j = cand[np.argmin(perp[cand])]
        if perp[j] < cfg.hole_dist and t0 >= hole_frame[j] - cfg.margin:
            keep.append(a)
        else:
            drop.append(a)
    print(f"[secuencia] {len(tracks)} -> {len(keep)} flyrocks validos "
          f"({len(drop)} descartados por origen/tiempo de tiro)")

    # --- visualizaciones ---
    gray_bg = cv2.cvtColor(
        (np.clip(gray / (gray.max() + 1e-6), 0, 1) ** 0.5 * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR)
    thick = 3 if w > 2500 else 2

    # A) validar alineacion: tiros coloreados por frame de detonacion sobre gris
    align_img = gray_bg.copy()
    fr = hole_frame
    fr_n = ((fr - fr.min()) / (np.ptp(fr) + 1e-6) * 255).astype(np.uint8)
    for (x, y), c in zip(Hpx, fr_n):
        col = cv2.applyColorMap(np.uint8([[c]]), cv2.COLORMAP_JET)[0, 0]
        cv2.circle(align_img, (int(x), int(y)), 7,
                   (int(col[0]), int(col[1]), int(col[2])), -1)
    cv2.imwrite(str(SEQ_DIR / "alineacion_tiros.png"), align_img)

    # B) trayectorias validas sobre gris
    over = gray_bg.copy()
    for a in drop:
        poly = a[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(over, [poly], False, (45, 45, 45), 1, cv2.LINE_AA)
    for i, a in enumerate(keep):
        c = cv2.applyColorMap(np.uint8([[(i * 41) % 256]]), cv2.COLORMAP_HSV)[0, 0]
        poly = a[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(over, [poly], False, (int(c[0]), int(c[1]), int(c[2])),
                      thick, cv2.LINE_AA)
    cv2.imwrite(str(SEQ_DIR / "trayectorias_validas_sobre_gris.png"), over)
    print(f"[viz] -> {SEQ_DIR}")


if __name__ == "__main__":
    main()
