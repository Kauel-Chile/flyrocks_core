"""
Polar v2: excluye el area de tronadura (polvo) via mapa de persistencia y
libera el umbral de energia -> recupera las rocas tenues de AFUERA.

Salidas (debug/out/):
  mask_gray.png          -> mascara de estelas generada (escala de grises)
  mask_over_gray.png     -> trayectorias de color superpuestas sobre la gris (fija)
  video_over_real.mp4    -> trayectorias dibujandose sobre el video real
  video_over_gray.mp4    -> trayectorias dibujandose sobre la mascara gris fija
  polar_result.npz       -> tracks [id,x,y,t] + origen (reutilizable)

Uso 1080p:  uv run python debug/polar_v2.py
Uso 4K:     uv run python debug/polar_v2.py --clip <clip_4k.mp4> \
                --dilate 30 --lateral 44 --growth 50 --eps 8
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
from utils.nodes.trajectory_analysis import DBSCANClusteringNode      # noqa: E402
from polar_tracker import polar_track                                 # noqa: E402
from cv_tracker import cv_track                                       # noqa: E402

C2_DIR = OUT_DIR / "2_comparacion_camino2"
CACHE_DIR = OUT_DIR / "_cache"
C2_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_paint(path, w, h):
    """Devuelve mascaras booleanas red/green/blue desde el PNG pintado."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"No pude leer el pintado: {path}")
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    red = (r > 100) & (r > g + 50) & (r > b + 50)
    green = (g > 100) & (g > r + 50) & (g > b + 50)
    blue = (b > 100) & (b > r + 50) & (b > g + 50)
    return red, green, blue


def build_persistence(tensor, w, h):
    pers = np.zeros((h, w), dtype=np.int32)
    xs = np.clip(tensor[:, 0].astype(int), 0, w - 1)
    ys = np.clip(tensor[:, 1].astype(int), 0, h - 1)
    np.add.at(pers, (ys, xs), 1)
    return pers


def build_gray_accumulator(tensor, w, h, gamma=0.5):
    """Mascara de estelas: maxima intensidad por pixel (estilo mascara cliente)."""
    acc = np.zeros((h, w), dtype=np.float32)
    xs = np.clip(tensor[:, 0].astype(int), 0, w - 1)
    ys = np.clip(tensor[:, 1].astype(int), 0, h - 1)
    np.maximum.at(acc, (ys, xs), tensor[:, 3])
    norm = (np.clip(acc / (acc.max() + 1e-6), 0, 1) ** gamma * 255).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def straightness(arr):
    p = arr[:, :2]
    net = np.hypot(*(p[-1] - p[0]))
    seg = np.diff(p, axis=0)
    path = np.hypot(seg[:, 0], seg[:, 1]).sum()
    return net / (path + 1e-6)


def detect_blast_start(tensor, baseline_end=28):
    """Frame donde arranca la tronadura: primer salto grande de eventos sobre
    la linea base (deriva del dron pre-tronadura)."""
    counts = np.bincount(tensor[:, 2].astype(int))
    base = np.median(counts[2:baseline_end][counts[2:baseline_end] > 0])
    thr = max(5 * base, base + 2000)
    over = np.where(counts > thr)[0]
    return int(over[0]) if len(over) else 0


def filter_tracks(tracks, min_pts, min_growth, min_straight, start_t=0,
                  blue_mask=None):
    keep = []
    for tr in tracks:
        arr = np.array(tr.pts)
        if len(arr) < 2:
            continue
        # azul = critica -> relaja geometria (no se pierde)
        in_blue = False
        if blue_mask is not None:
            px = np.clip(arr[:, 0].astype(int), 0, blue_mask.shape[1] - 1)
            py = np.clip(arr[:, 1].astype(int), 0, blue_mask.shape[0] - 1)
            in_blue = bool(blue_mask[py, px].any())
        mp = 3 if in_blue else min_pts
        mg = 30 if in_blue else min_growth
        ms = 0.4 if in_blue else min_straight
        if len(arr) < mp:
            continue
        if arr[0, 2] < start_t:          # nacio antes de la tronadura (fisica)
            continue
        grow = np.hypot(arr[:, 0] - arr[0, 0], arr[:, 1] - arr[0, 1]).max()
        if grow < mg:
            continue
        if straightness(arr) < ms:
            continue
        keep.append(arr)
    return keep


def color_for(i):
    c = cv2.applyColorMap(np.uint8([[(i * 41) % 256]]), cv2.COLORMAP_HSV)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def draw_full(bg, keep, origin, thick):
    """Dibuja todas las trayectorias completas sobre bg (copia)."""
    img = bg.copy()
    for i, arr in enumerate(keep):
        poly = arr[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], False, color_for(i), thick, cv2.LINE_AA)
    cv2.circle(img, tuple(origin.astype(int)), 12, (255, 255, 255), 2)
    return img


def draw_progressive(bg_provider, keep, origin, blast_cont, meta, out_mp4,
                     n_frames, fps, thick):
    """Video: dibuja las trazas incrementalmente. bg_provider(fidx)->frame BGR."""
    w, h = int(meta[0]), int(meta[1])
    out = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    cols = [color_for(i) for i in range(len(keep))]
    for fidx in range(n_frames):
        frame = bg_provider(fidx)
        if frame is None:
            break
        if blast_cont is not None:
            cv2.drawContours(frame, blast_cont, -1, (0, 0, 255), 2)
        for i, arr in enumerate(keep):
            past = arr[arr[:, 2] <= fidx + 2]
            if len(past) > 1:
                poly = past[:, :2].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [poly], False, cols[i], thick)
                hx, hy = poly[-1][0]
                cv2.circle(frame, (hx, hy), thick + 2, cols[i], -1)
        cv2.putText(frame, f"Frame {fidx} | flyrocks: {len(keep)}", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        out.write(frame)
    out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=str, default=str(CLIP_PATH))
    ap.add_argument("--paint", type=str, default=None,
                    help="PNG pintado (rojo/verde/azul). Si se da, el ROI es la "
                         "anotacion: extraer solo en rojo|azul, nunca en verde")
    ap.add_argument("--persist", type=int, default=6)
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--energy", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--tracker", choices=["polar", "cv"], default="cv")
    ap.add_argument("--lateral", type=float, default=22.0)
    ap.add_argument("--cross", type=float, default=18.0,
                    help="[cv] tolerancia perpendicular (px)")
    ap.add_argument("--max-speed", type=float, default=220.0,
                    help="[cv] velocidad maxima fisica (px/frame)")
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--growth", type=float, default=25.0)
    ap.add_argument("--straight", type=float, default=0.6)
    ap.add_argument("--start-frame", type=int, default=-1,
                    help="frame de inicio de tronadura (-1 = auto-detectar)")
    ap.add_argument("--no-video", action="store_true",
                    help="salta el render de videos (iteracion rapida, solo PNG)")
    cfg = ap.parse_args()
    clip = Path(cfg.clip)

    tensor, meta = extract_events(clip, stabilize=True)
    w, h = int(meta[0]), int(meta[1])
    thick = 3 if w > 2500 else 2

    pers = build_persistence(tensor, w, h)
    blast = (pers >= cfg.persist).astype(np.uint8) * 255
    k = cfg.dilate
    blast = cv2.dilate(blast, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys, xs = np.nonzero(pers)
    wgt = pers[ys, xs]
    origin = np.array([np.average(xs, weights=wgt), np.average(ys, weights=wgt)])
    print(f"[{w}x{h}] area polvo: {int((blast>0).sum()):,} px | "
          f"origen ({origin[0]:.0f}, {origin[1]:.0f})")

    ex = np.clip(tensor[:, 0].astype(int), 0, w - 1)
    ey = np.clip(tensor[:, 1].astype(int), 0, h - 1)
    blue_mask = None
    if cfg.paint:
        red, green, blue = parse_paint(cfg.paint, w, h)
        blue_mask = blue
        # rojo = extraer pero SIN el polvo persistente; azul = critico (ignora polvo)
        keep_mask = blue | (red & ~green & (blast == 0))
        t_out = tensor[keep_mask[ey, ex]]
        print(f"[paint] ROI (rojo sin-polvo | azul): {int(keep_mask.sum()):,} px "
              f"| eventos {len(tensor):,} -> {len(t_out):,}")
    else:
        outside = blast[ey, ex] == 0
        t_out = tensor[outside]
        print(f"Eventos: {len(tensor):,} -> {len(t_out):,} afuera")
    if cfg.energy > 0:
        t_out = t_out[t_out[:, 3] >= np.percentile(t_out[:, 3], cfg.energy)]

    # --- inicio de tronadura (filtro temporal: no hay flyrock antes) ---
    start_t = cfg.start_frame if cfg.start_frame >= 0 else detect_blast_start(tensor)
    seq = ROOT / "debug" / "Secuencia (2).csv"
    if seq.exists():
        import csv
        times = [float(r[4]) for r in csv.reader(open(seq)) if len(r) >= 5
                 and r[4].strip().replace('.', '').isdigit()]
        if times:
            dur_f = (max(times) - min(times)) / 1000 * 30
            print(f"[secuencia] {len(times)} tiros | detonacion dura "
                  f"{max(times)-min(times):.0f}ms ~ {dur_f:.0f} frames")
    print(f"[temporal] inicio de tronadura = frame {start_t} "
          f"(se descartan trazas nacidas antes)")

    ctx = DBSCANClusteringNode(eps=cfg.eps).run({"tensor_raw": t_out})
    if cfg.tracker == "cv":
        tracks = cv_track(ctx["unique_frames"], ctx["detections_by_frame"],
                          origin, cross_tol=cfg.cross, max_speed=cfg.max_speed)
    else:
        tracks = polar_track(ctx["unique_frames"], ctx["detections_by_frame"],
                             origin, cfg.lateral, patience=8)
    keep = filter_tracks(tracks, cfg.min_pts, cfg.growth, cfg.straight, start_t,
                         blue_mask=blue_mask)
    print(f"[{cfg.tracker}] Tracks: {len(tracks)} -> {len(keep)} flyrocks")

    # --- guardar tracks reutilizables ---
    rows = [np.column_stack([np.full(len(a), i), a]) for i, a in enumerate(keep)]
    if rows:
        np.savez_compressed(CACHE_DIR / "polar_result.npz",
                            traj=np.vstack(rows), origin=origin, meta=np.array(meta))

    # --- fondos y contorno ---
    gray_bg = build_gray_accumulator(tensor, w, h)
    cont, _ = cv2.findContours(blast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1) imagen fija: trazas de color sobre la gris
    over = draw_full(gray_bg, keep, origin, thick)
    cv2.drawContours(over, cont, -1, (0, 0, 255), 1)
    cv2.imwrite(str(C2_DIR / "trayectorias_sobre_gris.png"), over)

    if not cfg.no_video:
        # 2) video progresivo sobre la gris fija
        draw_progressive(lambda f: gray_bg.copy(), keep, origin, cont, meta,
                         C2_DIR / "video_sobre_gris.mp4", int(meta[2]) - 2, 30, thick)
        # 3) video progresivo sobre el video real
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS)

        def real_bg(_):
            ret, fr = cap.read()
            return fr if ret else None

        draw_progressive(real_bg, keep, origin, cont, meta,
                         C2_DIR / "video_sobre_real.mp4", int(meta[2]), fps, thick)
        cap.release()
    print(f"[viz] -> {C2_DIR}")


if __name__ == "__main__":
    main()
