"""
Tracker de VELOCIDAD CONSTANTE con continuidad fisica temporal (Camino 2).

Arregla los "teletransportes": una roca tiene velocidad finita y no invierte
direccion. El gate es una ELIPSE alineada al movimiento:
  - a-lo-largo del movimiento: tolerante (la rapidez varia)
  - perpendicular: angosto (NO salta a la estela vecina)
Ademas: tope de velocidad maxima, sin saltos temporales grandes, sin reversiones.

El prior radial (desde el origen de tronadura) solo inicializa la direccion del
primer paso; despues manda la velocidad estimada de la propia roca.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class CVTrack:
    __slots__ = ("pts", "pos", "vel", "last_t")

    def __init__(self, x, y, t):
        self.pts = [(x, y, t)]
        self.pos = np.array([x, y], float)
        self.vel = None          # se estima al 2do punto
        self.last_t = t

    def predict(self, t):
        if self.vel is None:
            return self.pos
        return self.pos + self.vel * (t - self.last_t)

    def add(self, x, y, t):
        new = np.array([x, y], float)
        dt = t - self.last_t
        v = (new - self.pos) / max(dt, 1)
        if self.vel is None:
            self.vel = v
        else:
            self.vel = 0.5 * self.vel + 0.5 * v   # suavizado
        self.pts.append((x, y, t))
        self.pos, self.last_t = new, t


def cv_track(frames, detections_by_frame, origin,
             cross_tol=18.0, max_speed=220.0, max_gap=3,
             along_frac=0.6, along_base=30.0, max_turn_deg=30.0):
    """Devuelve lista de CVTrack. Impone continuidad fisica temporal."""
    max_turn = np.deg2rad(max_turn_deg)
    active, done = [], []

    for t in frames:
        dets = detections_by_frame[t]
        if len(dets) == 0:
            continue

        if active:
            cost = np.full((len(active), len(dets)), 1e6)
            for i, tr in enumerate(active):
                dt = t - tr.last_t
                if dt > max_gap:
                    continue
                pred = tr.predict(t)
                off = dets - pred                      # (N,2) offset vs prediccion
                dist = np.hypot(off[:, 0], off[:, 1])

                if tr.vel is None:
                    # 1er enlace: direccion esperada = radial desde el origen
                    d = tr.pos - origin
                    speed_ref = np.hypot(*d)
                else:
                    d = tr.vel
                    speed_ref = np.hypot(*d) * dt
                nrm = np.hypot(*d)
                if nrm < 1e-6:
                    continue
                ux, uy = d / nrm                        # direccion unitaria
                along = off[:, 0] * ux + off[:, 1] * uy
                cross = np.abs(-off[:, 0] * uy + off[:, 1] * ux)

                # medir el paso real desde la posicion actual (no la prediccion)
                step = np.hypot(dets[:, 0] - tr.pos[0], dets[:, 1] - tr.pos[1]) / dt
                along_win = along_frac * max(speed_ref, 20) + along_base

                ok = (cross < cross_tol) & (along > -along_base) & \
                     (np.abs(along) < along_win) & (step < max_speed) & \
                     (dist < max_speed * dt)
                # prohibir reversion de direccion (giro > max_turn)
                if tr.vel is not None:
                    to_det = dets - tr.pos
                    ang = np.arctan2(to_det[:, 1], to_det[:, 0])
                    vang = np.arctan2(tr.vel[1], tr.vel[0])
                    turn = np.abs(np.arctan2(np.sin(ang - vang), np.cos(ang - vang)))
                    ok &= turn < max_turn
                cost[i][ok] = cross[ok] + 0.3 * np.abs(along[ok])

            row, col = linear_sum_assignment(cost)
            matched = set()
            for i, j in zip(row, col):
                if cost[i, j] < 1e5:
                    active[i].add(dets[j, 0], dets[j, 1], t)
                    matched.add(j)
            new_dets = [j for j in range(len(dets)) if j not in matched]
        else:
            new_dets = list(range(len(dets)))

        for j in new_dets:
            active.append(CVTrack(dets[j, 0], dets[j, 1], t))

        still = []
        for tr in active:
            if t - tr.last_t > max_gap:
                done.append(tr)
            else:
                still.append(tr)
        active = still

    return done + active
