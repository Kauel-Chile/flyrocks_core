"""
Prototipo de tracker POLAR (approach nuevo para vista cenital).

En vez de Kalman+Hungaro cartesiano con max_dist chico (que rompe las rocas
rapidas), trackeamos en coordenadas polares alrededor del origen de tronadura:
una roca = angulo theta ~constante + radio r creciente. Asociar por "corredor
angular" es robusto a saltos radiales grandes -> captura las rocas rapidas.

    uv run python debug/polar_tracker.py [--energy 90] [--lateral 25]
"""
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "debug"))
from harness import extract_events, CLIP_PATH, OUT_DIR                # noqa: E402
from utils.nodes.trajectory_analysis import (                        # noqa: E402
    EnergyPercentileFilterNode, DBSCANClusteringNode,
)


def get_detections(energy_pct):
    tensor, meta = extract_events(CLIP_PATH, stabilize=True)
    ctx = {"tensor_raw": tensor}
    ctx = EnergyPercentileFilterNode(percentile=energy_pct).run(ctx)
    ctx = DBSCANClusteringNode(eps=4.0).run(ctx)
    return ctx["unique_frames"], ctx["detections_by_frame"], meta


def estimate_origin(detections_by_frame, frames):
    """Origen ~ mediana de las detecciones de los primeros frames activos."""
    early = frames[: max(6, len(frames) // 5)]
    pts = np.vstack([detections_by_frame[t] for t in early
                     if len(detections_by_frame[t])])
    return np.median(pts, axis=0)


class Track:
    __slots__ = ("pts", "last_r", "last_t", "v_r", "theta")

    def __init__(self, x, y, t, r, th):
        self.pts = [(x, y, t)]
        self.last_r = r
        self.last_t = t
        self.v_r = None          # velocidad radial (px/frame), se estima al 2do punto
        self.theta = th

    def predict_r(self, t):
        if self.v_r is None:
            return self.last_r
        return self.last_r + self.v_r * (t - self.last_t)

    def add(self, x, y, t, r, th):
        if self.v_r is None and t > self.last_t:
            self.v_r = (r - self.last_r) / (t - self.last_t)
        elif t > self.last_t:
            # suavizado exponencial de la velocidad radial
            self.v_r = 0.6 * self.v_r + 0.4 * (r - self.last_r) / (t - self.last_t)
        self.pts.append((x, y, t))
        self.last_r, self.last_t, self.theta = r, t, th


def polar_track(frames, detections_by_frame, origin, lateral_tol, patience):
    """Devuelve lista de tracks. Asocia por corredor angular + continuidad en r."""
    active, done = [], []
    for t in frames:
        dets = detections_by_frame[t]
        if len(dets) == 0:
            continue
        dx = dets[:, 0] - origin[0]
        dy = dets[:, 1] - origin[1]
        r = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)

        if active:
            # matriz de costo: desviacion lateral (px) + residuo radial (px)
            cost = np.full((len(active), len(dets)), 1e6)
            for i, tr in enumerate(active):
                dth = np.abs(np.arctan2(np.sin(th - tr.theta),
                                        np.cos(th - tr.theta)))
                lateral = dth * r                      # px fuera del corredor
                r_pred = tr.predict_r(t)
                radial = np.abs(r - r_pred)
                # ventana radial generosa (rocas rapidas saltan mucho)
                r_win = 60 + (abs(tr.v_r) * (t - tr.last_t) if tr.v_r else 40)
                ok = (lateral < lateral_tol) & (radial < r_win) & (r >= tr.last_r - 5)
                cost[i][ok] = lateral[ok] + 0.5 * radial[ok]
            row, col = linear_sum_assignment(cost)
            matched_d = set()
            for i, j in zip(row, col):
                if cost[i, j] < 1e5:
                    active[i].add(dets[j, 0], dets[j, 1], t, r[j], th[j])
                    matched_d.add(j)
            new_dets = [j for j in range(len(dets)) if j not in matched_d]
        else:
            new_dets = list(range(len(dets)))

        for j in new_dets:
            active.append(Track(dets[j, 0], dets[j, 1], t, r[j], th[j]))

        still = []
        for tr in active:
            if t - tr.last_t > patience:
                done.append(tr)
            else:
                still.append(tr)
        active = still
    return done + active


def filter_tracks(tracks, min_pts=4, min_r_growth=40, max_theta_spread=0.25):
    keep = []
    for tr in tracks:
        if len(tr.pts) < min_pts:
            continue
        arr = np.array(tr.pts)
        r = np.hypot(arr[:, 0] - tr.pts[0][0], arr[:, 1] - tr.pts[0][1])
        if r.max() < min_r_growth:
            continue
        keep.append(arr)
    return keep


def render(keep, origin, meta, out_png, clip, out_mp4):
    w, h = int(meta[0]), int(meta[1])
    canvas = np.zeros((h, w, 3), np.uint8)
    for i, arr in enumerate(keep):
        c = cv2.applyColorMap(np.uint8([[(i * 41) % 256]]), cv2.COLORMAP_HSV)[0, 0]
        poly = arr[:, :2].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, (int(c[0]), int(c[1]), int(c[2])),
                      1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(origin.astype(int)), 10, (255, 255, 255), 2)
    cv2.imwrite(str(out_png), canvas)

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    cols = [cv2.applyColorMap(np.uint8([[(i * 41) % 256]]),
                              cv2.COLORMAP_HSV)[0, 0] for i in range(len(keep))]
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for i, arr in enumerate(keep):
            past = arr[arr[:, 2] <= fidx + 2]
            if len(past) > 1:
                c = cols[i]
                poly = past[:, :2].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [poly], False,
                              (int(c[0]), int(c[1]), int(c[2])), 2)
        cv2.putText(frame, f"Frame {fidx} | flyrocks(polar): {len(keep)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out.write(frame)
        fidx += 1
    cap.release()
    out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=90.0)
    ap.add_argument("--lateral", type=float, default=25.0,
                    help="tolerancia lateral al corredor radial (px)")
    ap.add_argument("--patience", type=int, default=8)
    cfg = ap.parse_args()

    frames, dets, meta = get_detections(cfg.energy)
    origin = estimate_origin(dets, frames)
    print(f"Origen: ({origin[0]:.0f}, {origin[1]:.0f})")
    tracks = polar_track(frames, dets, origin, cfg.lateral, cfg.patience)
    keep = filter_tracks(tracks)
    print(f"Tracks polares: {len(tracks)} -> {len(keep)} flyrocks (filtrados)")
    render(keep, origin, meta, OUT_DIR / "mask_polar.png",
           CLIP_PATH, OUT_DIR / "overlay_polar.mp4")
    print("[viz] mask_polar.png + overlay_polar.mp4")


if __name__ == "__main__":
    main()
