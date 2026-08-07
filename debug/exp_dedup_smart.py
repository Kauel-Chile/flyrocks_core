"""
Dedup INTELIGENTE: fusiona solo DUPLICADOS reales (misma linea + se solapan en
radio), en vez de colapsar todo un sector angular. Conserva rocas distintas
(incluidas parabolas) que el dedup grueso perdia.

Firma de linea por traza: (direccion, distancia perpendicular al origen con
signo, rango de radio). Dos trazas son duplicadas si direccion ~igual, perp
~igual y sus radios se solapan. Union-find para agrupar; deja la mas larga.
"""
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from clean_and_stitch import load_tracks, render

OUT = ROOT / "debug" / "out" / "6_mascaras" / "trayectorias"
GRAY = ROOT / "debug" / "out" / "6_mascaras" / "1_intensidad.png"
DTHETA, DPERP = 0.03, 30.0        # tolerancias (mas chico = fusiona menos)


def sig(a, o):
    p0, p1 = a[0, :2], a[-1, :2]
    d = p1 - p0
    n = np.hypot(*d)
    if n < 1:
        return None
    u = d / n
    perp = (o[0] - p0[0]) * u[1] - (o[1] - p0[1]) * u[0]   # perp con signo
    r = np.hypot(a[:, 0] - o[0], a[:, 1] - o[1])
    return np.arctan2(u[1], u[0]), perp, r.min(), r.max()


def main():
    tracks, origin = load_tracks()
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
        for a_ in range(len(idxs)):
            i = idxs[a_]; si = S[i]
            for j in cand:
                if j <= i:
                    continue
                sj = S[j]
                dang = abs(np.arctan2(np.sin(si[0] - sj[0]), np.cos(si[0] - sj[0])))
                overlap = max(si[2], sj[2]) < min(si[3], sj[3])
                if dang < DTHETA and abs(si[1] - sj[1]) < DPERP and overlap:
                    par[find(i)] = find(j)

    clusters = defaultdict(list)
    for i in range(len(tracks)):
        if S[i] is not None:
            clusters[find(i)].append(i)

    def extent(a):
        return np.hypot(a[-1, 0] - a[0, 0], a[-1, 1] - a[0, 1])

    kept = [max((tracks[i] for i in c), key=extent) for c in clusters.values()]
    bg = cv2.imread(str(GRAY))
    render(kept, origin, OUT / "exp_dedup_smart.png", bg)
    print(f"dedup inteligente: {len(tracks)} -> {len(kept)} (vs dedup grueso=174)")


if __name__ == "__main__":
    main()
