"""
Dedup v3 — usa TIEMPO + AZUL (info que el dedup viejo ignoraba).

Reglas:
 1) AZUL = UNA trayectoria. Todas las trazas que tocan un mismo trazo azul son
    la MISMA roca -> se colapsan a UNA, y esa queda garantizada.
 2) Resto: se agrupa por sector angular PERO se sub-agrupa por TIEMPO.
    - mismo sector + tiempos que SE SOLAPAN  -> duplicados -> se colapsan
    - mismo sector + tiempos DISJUNTOS       -> rocas DISTINTAS -> se conservan
    (el dedup viejo colapsaba todo el sector => perdia parabolas)
 3) La intensidad NO decide (varia por angulo); solo se usa el largo para elegir
    el representante.

    uv run python debug/exp_dedup_v3.py [--da 0.03]
"""
import sys
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from clean_and_stitch import load_tracks, render                     # noqa: E402
from paint_filter import PaintMask                                   # noqa: E402

OUT = ROOT / "debug" / "out" / "6_mascaras" / "trayectorias"
GRAY = ROOT / "debug" / "out" / "6_mascaras" / "1_intensidad.png"
PAINT = ROOT / "debug" / "out" / "4_fase0_referencias" / "lienzo_para_pintar_gris - Copy.png"


def extent(a):
    return np.hypot(a[-1, 0] - a[0, 0], a[-1, 1] - a[0, 1])


def radial_angle(a, o):
    c = a[:, :2].mean(0)
    return np.arctan2(c[1] - o[1], c[0] - o[0])


def _fit_curve(P):
    """Ajusta curva suave (parabola) a un set de puntos. -> (c,u,v,poly)"""
    c = P.mean(0); d = P - c
    w_, V = np.linalg.eigh(d.T @ d)
    u = V[:, int(np.argmax(w_))]
    v = np.array([-u[1], u[0]])
    ta = d @ u
    deg = 2 if len(P) >= 5 else 1
    return c, u, v, np.poly1d(np.polyfit(ta, d @ v, deg))


def _resid(track, model):
    """Residuo mediano de un segmento respecto a la curva."""
    c, u, v, poly = model
    d = track[:, :2] - c
    return float(np.median(np.abs((d @ v) - poly(d @ u))))


def ransac_dominant(members, tracks, min_span, tol=30.0, iters=120, max_out=8):
    """Encuentra iterativamente las curvas dominantes: la curva con la que mas
    segmentos concuerdan; sus inliers forman UNA trayectoria. Luego se repite
    con los sobrantes (asi un area con varios trazos pegados da varias)."""
    import random
    remaining = list(members)
    out = []
    while len(remaining) >= 2 and len(out) < max_out:
        best_inl, best_score = None, 0.0
        for _ in range(iters):
            s = random.sample(remaining, min(2, len(remaining)))
            P = np.vstack([tracks[i][:, :2] for i in s])
            if len(P) < 4:
                continue
            try:
                model = _fit_curve(P)
            except Exception:
                continue
            inl = [i for i in remaining if _resid(tracks[i], model) < tol]
            if len(inl) < 2:
                continue
            pts = np.vstack([tracks[i][:, :2] for i in inl])
            span = np.hypot(*(pts.max(0) - pts.min(0)))     # cobertura
            score = span * np.sqrt(len(inl))
            if score > best_score:
                best_score, best_inl = score, inl
        if best_inl is None:
            break
        P = np.vstack([tracks[i] for i in best_inl])
        span = np.hypot(*(P[:, :2].max(0) - P[:, :2].min(0)))
        if span < min_span:
            break
        # curva final suave sobre los inliers
        c, u, v, poly = _fit_curve(P[:, :2])
        ta = (P[:, :2] - c) @ u
        tt = np.linspace(ta.min(), ta.max(), 40)
        curve = c + np.outer(tt, u) + np.outer(poly(tt), v)
        times = np.linspace(P[:, 2].min(), P[:, 2].max(), len(curve))
        out.append(np.column_stack([curve, times]))
        remaining = [i for i in remaining if i not in set(best_inl)]
    return out


def dedup_v3(tracks, origin, da, blue_lab):
    result, used, blue_curves = [], set(), []

    # --- 1) AZUL: cada trazo azul = exactamente UNA trayectoria ---
    # Una traza PERTENECE al azul solo si la MAYOR PARTE de ella esta dentro
    # (no si apenas lo cruza). Luego se FUSIONAN sus fragmentos ordenados por
    # TIEMPO -> reconstruye la curva que el usuario marco.
    # El trazo azul es un AREA que contiene proyecciones. Se buscan los segmentos
    # DENTRO y se ENCADENAN por continuidad espacio-temporal + direccion. Cada
    # cadena = una trayectoria (asi trazos que se tocan dan cadenas separadas).
    MIN_FRAC, MAX_DT, MAX_GAP, MAX_TURN = 0.5, 25, 300.0, np.deg2rad(45)
    MIN_SPAN = 120.0
    n_blue = 0
    if blue_lab is not None:
        h, w = blue_lab.shape
        for lab in range(1, blue_lab.max() + 1):
            comp = (blue_lab == lab)
            if comp.sum() < 20:
                continue
            members = []
            for i, a in enumerate(tracks):
                if i in used:
                    continue
                px = np.clip(a[:, 0].astype(int), 0, w - 1)
                py = np.clip(a[:, 1].astype(int), 0, h - 1)
                if comp[py, px].mean() >= MIN_FRAC:
                    members.append(i)
            if not members:
                continue

            def d_of(i):
                a = tracks[i]
                v = a[-1, :2] - a[0, :2]
                n = np.hypot(*v)
                return v / n if n > 1e-6 else np.array([1.0, 0.0])

            # --- PASO 1: fusionar duplicados paralelos dentro del area ---
            sg = {}
            for i in members:
                a = tracks[i]
                v = a[-1, :2] - a[0, :2]
                n = np.hypot(*v)
                if n < 1:
                    continue
                u = v / n
                perp = (origin[0]-a[0, 0])*u[1] - (origin[1]-a[0, 1])*u[0]
                r = np.hypot(a[:, 0]-origin[0], a[:, 1]-origin[1])
                sg[i] = (np.arctan2(u[1], u[0]), perp, r.min(), r.max())
            ks = list(sg)
            pp = {i: i for i in ks}

            def fnd(x):
                while pp[x] != x:
                    pp[x] = pp[pp[x]]; x = pp[x]
                return x

            for ii in range(len(ks)):
                si = sg[ks[ii]]
                for jj in range(ii + 1, len(ks)):
                    sj = sg[ks[jj]]
                    da_ = abs(np.arctan2(np.sin(si[0]-sj[0]), np.cos(si[0]-sj[0])))
                    if da_ < 0.06 and abs(si[1]-sj[1]) < 45 \
                            and max(si[2], sj[2]) < min(si[3], sj[3]):
                        pp[fnd(ks[ii])] = fnd(ks[jj])
            grp = defaultdict(list)
            for i in ks:
                grp[fnd(i)].append(i)
            ext_i = lambda i: np.hypot(*(tracks[i][-1, :2] - tracks[i][0, :2]))
            reps = [max(g, key=ext_i) for g in grp.values()]
            n_dup = len(members) - len(reps)
            members = reps
            members.sort(key=lambda i: tracks[i][:, 2].min())
            # --- PASO 2: RANSAC de curva dominante (iterativo) ---
            chains = ransac_dominant(members, tracks, MIN_SPAN)
            for ch in chains:
                result.append(ch); blue_curves.append(ch)
            used.update(members)
            n_blue += len(chains)
            print(f"   azul#{lab}: {len(members)+n_dup} segmentos -> "
                  f"{len(members)} tras fusionar duplicados -> "
                  f"{len(chains)} trayectoria(s)")

    # --- 2) Resto: sector angular + sub-grupo por TIEMPO ---
    rest = [i for i in range(len(tracks)) if i not in used]
    buckets = defaultdict(list)
    for i in rest:
        buckets[round(radial_angle(tracks[i], origin) / da)].append(i)

    for idxs in buckets.values():
        idxs.sort(key=lambda i: tracks[i][:, 2].min())
        groups = []
        for i in idxs:
            t0, t1 = tracks[i][:, 2].min(), tracks[i][:, 2].max()
            for g in groups:
                if not (t1 < g["t0"] or t0 > g["t1"]):     # se solapan en tiempo
                    g["idx"].append(i)
                    g["t0"] = min(g["t0"], t0); g["t1"] = max(g["t1"], t1)
                    break
            else:
                groups.append({"idx": [i], "t0": t0, "t1": t1})
        for g in groups:
            result.append(tracks[max(g["idx"], key=lambda i: extent(tracks[i]))])
    return result, n_blue, blue_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da", type=float, default=0.03)
    cfg = ap.parse_args()

    tracks, origin = load_tracks()
    blue_lab = None
    pm = None
    if PAINT.exists():
        pm = PaintMask(PAINT)
        nlab, blue_lab = cv2.connectedComponents(pm.blue.astype(np.uint8))
        print(f"trazos azules detectados: {nlab-1}")

    kept, n_blue, blue_curves = dedup_v3(tracks, origin, cfg.da, blue_lab)
    print(f"dedup v3 (tiempo+azul): {len(tracks)} -> {len(kept)}   "
          f"[{n_blue} trazos azules con trayectoria]")

    bg = cv2.imread(str(GRAY))
    render(kept, origin, OUT / "exp_dedup_v3.png", bg)

    # --- DIAGNOSTICO: trazos azules + todo en gris + reconstruidas resaltadas ---
    diag = bg.copy()
    if pm is not None:
        diag[pm.blue] = (255, 120, 0)                       # tus trazos (azul)
    for a in kept:                                          # todas, gris tenue
        cv2.polylines(diag, [a[:, :2].astype(np.int32).reshape(-1, 1, 2)],
                      False, (70, 70, 70), 2)
    for a in blue_curves:                                   # reconstruidas: amarillo grueso
        cv2.polylines(diag, [a[:, :2].astype(np.int32).reshape(-1, 1, 2)],
                      False, (0, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(diag, "azul=lo que pintaste   amarillo=trayectoria reconstruida",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.imwrite(str(OUT / "exp_dedup_v3_azul.png"), diag)
    print(f"-> {OUT / 'exp_dedup_v3_azul.png'}")


if __name__ == "__main__":
    main()
