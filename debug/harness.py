"""
Harness de debug para el pipeline de Flyrocks.

Objetivo: correr UNA vez las etapas caras (estabilizacion -> extraccion ->
clustering -> tracking) sobre un clip real, CACHEAR las trayectorias crudas a
disco, y luego iterar filtros/visualizaciones en milisegundos.

Uso:
    uv run python debug/harness.py            # corre (usa cache si existe)
    uv run python debug/harness.py --force     # recomputa la etapa cara
    uv run python debug/harness.py --no-stab   # sin compensacion de movimiento

Salidas (en debug/out/):
    cache_raw.npz     -> tensor de trayectorias crudas [id, x, y, t]
    overlay_raw.mp4   -> el clip con TODAS las trayectorias crudas superpuestas
    mask_raw.png      -> mascara: todas las polilineas sobre fondo negro
"""
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

# --- hacer importables los nodos reales de src/ ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.nodes.trajectory_analysis import (  # noqa: E402
    EnergyPercentileFilterNode, DBSCANClusteringNode,
    KalmanTrackerNode, TrajectoryCleanerNode,
)

# ======================================================================
# CONFIG
# ======================================================================
SCRATCH = Path(
    r"C:\Users\carlo\AppData\Local\Temp\claude"
    r"\D--PROYECTOS-Enaex---Flyrocks-detovision-standalone-flyrocks-core"
    r"\3f0c6b88-01e1-4a43-9f40-09651754d37b\scratchpad"
)
CLIP_PATH = SCRATCH / "clip_blast.mp4"       # ventana de tronadura 1080p
OUT_DIR = ROOT / "debug" / "out"

NOISE_THRESHOLD = 8       # umbral de diferencia de intensidad
ENERGY_PCT = 96.0         # percentil de energia a conservar
DBSCAN_EPS = 5.0          # radio de clustering espacial
PATIENCE = 20             # max_lost_frames del tracker (fijo, sin grid search)


# ======================================================================
# EXTRACTOR CON COMPENSACION DE MOVIMIENTO (drone drift)
# ======================================================================
def estimate_global_motion(prev_gray, curr_gray):
    """Estima transform afin prev->curr usando features del terreno (RANSAC
    descarta las rocas/humo como outliers). Devuelve M 2x3 o None."""
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=600, qualityLevel=0.01, minDistance=20
    )
    if prev_pts is None or len(prev_pts) < 12:
        return None
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None
    )
    if curr_pts is None:
        return None
    good = status.ravel() == 1
    if good.sum() < 12:
        return None
    M, _ = cv2.estimateAffinePartial2D(
        prev_pts[good], curr_pts[good], method=cv2.RANSAC, ransacReprojThreshold=3
    )
    return M


def extract_events(clip_path, stabilize=True):
    """Frame-differencing con compensacion opcional de movimiento del dron.
    Devuelve tensor_raw [x, y, t, intensidad] y (w, h, n_frames)."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir {clip_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Clip vacio")
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    cloud = []
    events_per_frame = []
    frame_index = 2
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.GaussianBlur(
            cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (3, 3), 0
        )

        ref_gray = prev_gray
        if stabilize:
            M = estimate_global_motion(prev_gray, curr_gray)
            if M is not None:
                ref_gray = cv2.warpAffine(prev_gray, M, (w, h))

        diff = cv2.absdiff(ref_gray, curr_gray)
        ys, xs = np.nonzero(diff > NOISE_THRESHOLD)
        events_per_frame.append(len(xs))
        if ys.size > 0:
            inten = diff[ys, xs]
            ts = np.full(ys.size, frame_index, dtype=np.uint16)
            cloud.append(np.column_stack((xs, ys, ts, inten)))

        prev_gray = curr_gray
        frame_index += 1

    cap.release()
    tensor = np.concatenate(cloud, axis=0) if cloud else None
    epf = np.array(events_per_frame)
    print(f"  Frames procesados: {frame_index - 2} | eventos totales: "
          f"{0 if tensor is None else len(tensor):,}")
    print(f"  Eventos/frame  min={epf.min()} mediana={int(np.median(epf))} "
          f"max={epf.max()}  (mas alto = mas ruido residual)")
    return tensor, (w, h, frame_index)


# ======================================================================
# ETAPA CARA (cacheada)
# ======================================================================
def run_expensive_stage(clip_path, stabilize, force):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "cache_raw.npz"
    if cache.exists() and not force:
        print(f"[cache] Cargando {cache.name} (usa --force para recomputar)")
        d = np.load(cache)
        return d["traj"], tuple(d["meta"])

    print(f"[extract] Estabilizacion={'ON' if stabilize else 'OFF'}")
    t0 = time.time()
    tensor, meta = extract_events(clip_path, stabilize=stabilize)
    if tensor is None:
        raise RuntimeError("No se extrajeron eventos.")

    ctx = {"tensor_raw": tensor}
    ctx = EnergyPercentileFilterNode(percentile=ENERGY_PCT).run(ctx)
    ctx = DBSCANClusteringNode(eps=DBSCAN_EPS).run(ctx)
    ctx["optimal_patience"] = PATIENCE
    print(f"[track] Kalman + Hungaro (patience={PATIENCE})...")
    ctx = KalmanTrackerNode().run(ctx)
    ctx = TrajectoryCleanerNode().run(ctx)

    traj = ctx.get("final_trajectories")
    if traj is None:
        raise RuntimeError("Ninguna trayectoria sobrevivio al cleaner.")

    n_traj = len(np.unique(traj[:, 0]))
    print(f"[done] {n_traj} trayectorias crudas en {time.time() - t0:.1f}s")
    np.savez_compressed(cache, traj=traj, meta=np.array(meta))
    print(f"[cache] Guardado en {cache}")
    return traj, meta


# ======================================================================
# VISUALIZACION
# ======================================================================
def split_by_id(traj):
    """Devuelve dict {id: array [x,y,t] ordenado por t}."""
    out = {}
    for tid in np.unique(traj[:, 0]):
        pts = traj[traj[:, 0] == tid]
        pts = pts[np.argsort(pts[:, 3])]
        out[int(tid)] = pts[:, 1:4]
    return out


def color_for(i):
    rgba = cv2.applyColorMap(np.uint8([[(i * 37) % 256]]), cv2.COLORMAP_HSV)[0, 0]
    return int(rgba[0]), int(rgba[1]), int(rgba[2])


def render_overlay(traj, clip_path, out_path):
    tracks = split_by_id(traj)
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    colors = {tid: color_for(i) for i, tid in enumerate(tracks)}
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        on_screen = 0
        for tid, pts in tracks.items():
            past = pts[pts[:, 2] <= fidx + 2]
            if len(past) > 1:
                on_screen += 1
                poly = past[:, :2].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [poly], False, colors[tid], 2)
                hx, hy = poly[-1][0]
                cv2.circle(frame, (hx, hy), 3, colors[tid], -1)
        cv2.putText(frame, f"Frame {fidx} | trazas totales: {len(tracks)} | "
                    f"activas: {on_screen}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out.write(frame)
        fidx += 1
    cap.release()
    out.release()
    print(f"[viz] overlay -> {out_path}")


def render_mask(traj, meta, out_path):
    w, h = int(meta[0]), int(meta[1])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    tracks = split_by_id(traj)
    for i, (tid, pts) in enumerate(tracks.items()):
        poly = pts[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, color_for(i), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
    print(f"[viz] mascara -> {out_path}  ({len(tracks)} trazas)")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recomputa etapa cara")
    ap.add_argument("--no-stab", action="store_true", help="sin estabilizacion")
    args = ap.parse_args()

    traj, meta = run_expensive_stage(
        CLIP_PATH, stabilize=not args.no_stab, force=args.force
    )
    render_overlay(traj, CLIP_PATH, OUT_DIR / "overlay_raw.mp4")
    render_mask(traj, meta, OUT_DIR / "mask_raw.png")


if __name__ == "__main__":
    main()
