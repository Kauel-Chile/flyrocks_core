"""
Paso 2 (deduplicacion) y Paso 3 (costura de fragmentos) sobre los tracks
cacheados. Produce imagenes conservador vs completo para comparar visualmente.

Idea: las estelas son radiales desde el origen. Se agrupan los tracks por
ANGULO radial (origen -> centroide del track). Dentro de cada grupo (misma
estela):
  - DEDUP  : se deja el track mas largo (quita duplicados apilados).
  - COSTURA: se combinan TODOS los puntos del grupo, ordenados por radio, y se
             saca una linea limpia (mediana por tramo) -> une los fragmentos.

conservador = bin angular grande (fusiona mas, menos trayectorias, mas limpio)
completo    = bin angular fino  (fusiona menos, mas trayectorias, no pierde)

Salidas en out/5_dedup_costura/.
"""
import sys
import csv
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
CACHE = ROOT / "debug" / "out" / "_cache"
GRAY = ROOT / "debug" / "out" / "0_diagnostico" / "comparacion_4k" / "5_MI_gris_igualada_g085.png"
REALFRAME = ROOT / "debug" / "out" / "4_fase0_referencias" / "frame_full_res_13s.png"
PAINT = ROOT / "debug" / "out" / "4_fase0_referencias" / "lienzo_para_pintar_gris - Copy.png"
OUT = ROOT / "debug" / "out" / "5_dedup_costura"
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = {"conservador": 0.05, "completo": 0.02}   # tolerancia angular (rad)


def load_tracks():
    d = np.load(CACHE / "polar_result.npz")
    traj, origin = d["traj"], d["origin"]
    tracks = []
    for i in np.unique(traj[:, 0]):
        a = traj[traj[:, 0] == i][:, 1:4]
        tracks.append(a[np.argsort(a[:, 2])])
    return tracks, origin


def radial_angle(arr, origin):
    c = arr[:, :2].mean(0)
    return np.arctan2(c[1] - origin[1], c[0] - origin[0])


def extent(arr):
    return np.hypot(arr[-1, 0] - arr[0, 0], arr[-1, 1] - arr[0, 1])


def bin_by_angle(tracks, origin, da):
    groups = {}
    for arr in tracks:
        key = round(radial_angle(arr, origin) / da)
        groups.setdefault(key, []).append(arr)
    return groups


def dedup(tracks, origin, da):
    """Deja el track mas largo por sector angular."""
    out = []
    for grp in bin_by_angle(tracks, origin, da).values():
        out.append(max(grp, key=extent))
    return out


def smooth_line(line, iters=3):
    """Suaviza la polilinea (media movil), fijando los extremos."""
    out = line.astype(float).copy()
    for _ in range(iters):
        if len(out) > 2:
            out[1:-1] = (out[:-2] + out[1:-1] + out[2:]) / 3.0
    return out


def stitch(tracks, origin, da, slices=24):
    """Combina todos los fragmentos de un sector en una linea limpia y suave."""
    out = []
    for grp in bin_by_angle(tracks, origin, da).values():
        pts = np.vstack([g[:, :2] for g in grp])
        r = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
        o = np.argsort(r)
        pts, r = pts[o], r[o]
        if r[-1] - r[0] < 1:
            continue
        edges = np.linspace(r[0], r[-1], slices + 1)
        line = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (r >= a) & (r <= b)
            if m.any():
                line.append(np.median(pts[m], axis=0))
        if len(line) >= 2:
            out.append(smooth_line(np.array(line)))
    return out


def apply_H(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def load_shots():
    """Tiros en pixeles (homografia precisa) + su frame de detonacion.
    frame = 48 (primer tiro, t=14.10s) + (tiempo_ms - min)/1000 * 30."""
    hp = CACHE / "homografia.npz"
    if not hp.exists():
        return None, None
    H = np.load(hp)["H"]
    XY, T = [], []
    for r in list(csv.reader(open(ROOT / "debug" / "Secuencia (2).csv")))[1:]:
        if len(r) >= 5 and r[4].strip().replace('.', '').isdigit():
            XY.append((float(r[1]), float(r[2]))); T.append(float(r[4]))
    T = np.array(T)
    frames = 48 + (T - T.min()) / 1000 * 30
    return apply_H(H, np.array(XY)), frames


def extend_to_shot(tracks, shots_px, origin, max_back=1800, perp_tol=220):
    """Para cada traza (ordenada adentro->afuera) busca el tiro de origen mas
    cercano en su linea. Devuelve (traza_medida, tiro_o_None)."""
    out = []
    for arr in tracks:
        r = np.hypot(arr[:, 0] - origin[0], arr[:, 1] - origin[1])
        a = arr[np.argsort(r)]
        p_in, p_out = a[0, :2], a[-1, :2]
        d = p_in - p_out
        n = np.hypot(*d)
        shot = None
        if n > 1e-6 and shots_px is not None:
            vb = d / n
            rel = shots_px - p_in
            along = rel @ vb
            perp = np.hypot(*(rel - np.outer(along, vb)).T)
            ok = (along > 0) & (along < max_back) & (perp < perp_tol)
            if ok.any():
                shot = shots_px[np.where(ok)[0][np.argmin(perp[ok])]]
        out.append((a, shot))
    return out


def draw_dotted(img, p1, p2, col, gap=22, rad=3):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    n = max(int(np.hypot(*(p2 - p1)) / gap), 1)
    for i in range(n + 1):
        cv2.circle(img, tuple((p1 + (p2 - p1) * i / n).astype(int)), rad, col, -1)


def render_extended(items, shots_px, path, bg):
    canvas = bg.copy()
    if shots_px is not None:                       # grilla de tiros en rojo
        for x, y in shots_px:
            cv2.circle(canvas, (int(x), int(y)), 7, (0, 0, 255), -1)
    for i, (a, shot) in enumerate(items):
        col = color(i)
        poly = a[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, col, 3, cv2.LINE_AA)   # medido: solido
        if shot is not None:
            draw_dotted(canvas, shot, a[0, :2], col)                # inferido: punteado
    cv2.imwrite(str(path), canvas)


def fit_and_extend(tracks, shots_px, shot_frames, origin,
                   max_back=1600, perp_tol=340):
    """Ajusta cada traza a una parabola (en su eje principal), la suaviza, y
    PROLONGA la misma curva hasta el tiro de origen (siguiendo la curva).
    Elige el tiro por ESPACIO + TIEMPO: alineado/cercano y que YA haya detonado
    cuando la roca aparece. Devuelve (curva_medida, extension_o_None, tiro_o_None)."""
    out = []
    for arr in tracks:
        P = arr[:, :2].astype(float)
        birth = float(arr[:, 2].min()) if arr.shape[1] > 2 else None
        if len(P) < 3:
            out.append((P, None, None)); continue
        c = P.mean(0)
        d = P - c
        w, V = np.linalg.eigh(d.T @ d)
        u = V[:, int(np.argmax(w))]                 # eje principal (vuelo)
        t = d @ u
        r = np.hypot(P[:, 0] - origin[0], P[:, 1] - origin[1])
        if np.std(t) > 1e-6 and np.corrcoef(t, r)[0, 1] < 0:
            u = -u; t = d @ u                        # orienta t hacia AFUERA
        v = np.array([-u[1], u[0]])                  # perpendicular
        s = d @ v
        deg = 2 if len(P) >= 5 else 1
        poly = np.poly1d(np.polyfit(t, s, deg))
        tmin, tmax = t.min(), t.max()

        def curve(ta):
            return c + np.outer(ta, u) + np.outer(poly(ta), v)

        meas = curve(np.linspace(tmin, tmax, 40))    # medido suavizado a la parabola

        ext = shot = None
        if shots_px is not None:
            st = (shots_px - c) @ u
            ss = (shots_px - c) @ v
            behind = (st < tmin) & (st > tmin - max_back)
            if behind.any():
                cand = np.where(behind)[0]
                perp = np.abs(ss[cand] - poly(st[cand]))        # desvio de la curva
                cost = perp + 0.3 * (tmin - st[cand])           # espacio: alineado+cerca
                if birth is not None and shot_frames is not None:
                    sf = shot_frames[cand]
                    # penaliza fuerte si el tiro detona DESPUES de aparecer la roca
                    cost = cost + 2.5 * np.maximum(0, sf - birth) + 0.4 * np.abs(sf - birth)
                j = cand[int(np.argmin(cost))]
                if abs(ss[j] - poly(st[j])) < perp_tol:
                    shot = shots_px[j]
                    npt = max(int((tmin - st[j]) / 15), 3)
                    ext = curve(np.linspace(st[j], tmin, npt))  # MISMA parabola
        out.append((meas, ext, shot))
    return out


def render_fit(items, shots_px, path, bg):
    canvas = bg.copy()
    if shots_px is not None:
        for x, y in shots_px:
            cv2.circle(canvas, (int(x), int(y)), 8, (0, 0, 255), -1)
    for i, (meas, ext, shot) in enumerate(items):
        col = color(i)
        cv2.polylines(canvas, [meas.astype(np.int32).reshape(-1, 1, 2)],
                      False, col, 3, cv2.LINE_AA)
        if ext is not None:
            for p in ext:                            # extension = punteado
                cv2.circle(canvas, tuple(p.astype(int)), 3, col, -1)
    cv2.imwrite(str(path), canvas)


def color(i):
    c = cv2.applyColorMap(np.uint8([[(i * 41) % 256]]), cv2.COLORMAP_HSV)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def render(tracks, origin, path, bg):
    canvas = bg.copy()
    for i, a in enumerate(tracks):
        poly = a[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, color(i), 3, cv2.LINE_AA)
    cv2.circle(canvas, tuple(origin.astype(int)), 12, (255, 255, 255), 2)
    cv2.imwrite(str(path), canvas)


def blue_coverage(tracks, origin, bg):
    img = cv2.imread(str(PAINT))
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    blue = ((b > 100) & (b > r + 50) & (b > g + 50)).astype(np.uint8)
    n, labels = cv2.connectedComponents(blue)
    # marca de cobertura por track-point
    hit = np.zeros(n, bool)
    for a in tracks:
        px = np.clip(a[:, 0].astype(int), 0, blue.shape[1] - 1)
        py = np.clip(a[:, 1].astype(int), 0, blue.shape[0] - 1)
        for lab in np.unique(labels[py, px]):
            if lab > 0:
                hit[lab] = True
    canvas = bg.copy()
    matched = missed = 0
    for lab in range(1, n):
        mask = (labels == lab)
        col = (0, 255, 0) if hit[lab] else (0, 0, 255)
        matched += hit[lab]; missed += not hit[lab]
        canvas[mask] = col
    for i, a in enumerate(tracks):
        poly = a[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "azul_cobertura.png"), canvas)
    print(f"[azul] {matched}/{n-1} trazos azules con trayectoria "
          f"({missed} sin trayectoria = revisar)")


def main():
    tracks, origin = load_tracks()
    bg = cv2.imread(str(GRAY))
    if bg is None:
        bg = np.zeros((2160, 3840, 3), np.uint8)
    real = cv2.imread(str(REALFRAME))   # frame real del video (pozo completo)
    shots_px, shot_frames = load_shots()
    print(f"Tracks crudos: {len(tracks)}")
    render(tracks, origin, OUT / "0_crudo.png", bg)

    final_for_blue = None
    for name, da in PRESETS.items():
        d = dedup(tracks, origin, da)
        render(d, origin, OUT / f"1_dedup_{name}.png", bg)
        s = stitch(tracks, origin, da)
        render(s, origin, OUT / f"2_costura_{name}.png", bg)
        # ajuste a parabola + prolongacion al tiro por ESPACIO+TIEMPO (punteado)
        e = fit_and_extend(d, shots_px, shot_frames, origin)
        render_fit(e, shots_px, OUT / f"3_extendido_{name}.png", bg)
        if real is not None:
            render_fit(e, shots_px, OUT / f"3_extendido_{name}_sobre_video.png", real)
        n_ext = sum(1 for _, ex, _ in e if ex is not None)
        print(f"[{name}] dedup: {len(tracks)} -> {len(d)} | costura -> {len(s)} "
              f"| extendidas al tiro: {n_ext}/{len(e)}")
        if name == "completo":
            final_for_blue = [m for m, _, _ in e]
    blue_coverage(final_for_blue, origin, bg)
    print(f"[listo] imagenes en {OUT}")


if __name__ == "__main__":
    main()
