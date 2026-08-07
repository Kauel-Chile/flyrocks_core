"""
PASO 8 — MATCH DE TIROS: correspondencias tiro<->pixel y matriz H.

El calculo de la matriz ya existia (debug/homografia.py) y NO se cambia: se
reusa el mismo ajuste AFIN por minimos cuadrados con el mundo centrado, que
replica getSafeAffineMatrix del frontend. Lo que faltaba era la herramienta
para OBTENER las correspondencias, que hasta ahora estaban escritas a mano en
el codigo.

--------------------------------------------------------------------------
COMO SE USA (el ciclo completo toma menos de un minuto)
--------------------------------------------------------------------------
  1. Elige la imagen base con el trackbar "frame" (o tecla F: fondo promedio).
     Conviene un frame ANTES de la detonacion: los pozos se ven limpios.
  2. Click en un tiro del panel DERECHO (la malla del CSV)  -> queda ARMADO.
  3. Click en su pozo en el panel IZQUIERDO (el video)      -> correspondencia.
  4. Repite 3 veces. Al tercer punto aparece la malla completa proyectada.
  5. DESDE AHI ES AUTOMATICO: con la malla encima, basta con hacer click en un
     pozo del panel izquierdo (sin armar nada) y se asigna solo al tiro
     proyectado mas cercano. Agrega 2 o 3 mas y mira como baja el RMS.
  6. Tecla G para exportar.

Las cuatro anclas SUGERIDAS (circulo blanco en la malla) son los tiros mas
extremos: anclas muy juntas dan un ajuste mal condicionado que se ve bien en
el centro y se abre en los bordes.

--------------------------------------------------------------------------
CONTROLES
--------------------------------------------------------------------------
  click izq (panel malla)   armar tiro / desarmar si es el mismo
  click izq (panel frame)   fijar el pixel del tiro armado;
                            si no hay armado y ya hay H, se auto-asigna

  ZOOM (varias vias: la rueda depende del backend de OpenCV y en Windows a
  veces no llega, asi que NO es la unica)
  trackbar "zoom %"         la via infalible: 100 = cuadro completo, 2000 = 20x
  + / -                     acerca / aleja SOBRE EL CURSOR
  click der                 acerca ahi mismo   (Ctrl + click der = aleja)
  rueda                     acerca / aleja sobre el cursor, si tu build la pasa
  E                         encuadra automaticamente la malla proyectada
  R                         vuelve al cuadro completo
  boton medio + arrastrar   desplazar (o flechas del teclado)

  F                         alterna imagen base: frame de video / fondo promedio
  M                         muestra u oculta la malla proyectada
  U                         deshace la ultima correspondencia
  D                         borra la correspondencia del tiro armado
  G                         exporta H + correspondencias + validacion
  Q / ESC                   salir

    uv run python debug/pre_tiros.py
    uv run python debug/pre_tiros.py --seed     # parte con las 4 ya conocidas
    uv run python debug/pre_tiros.py --test     # sin ventana (verifica que corre)
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pre_common import CLIP, OUT, CACHE, START, label, save

D_TIROS = OUT / "05_tiros"
CSV_DEF = Path(__file__).resolve().parent / "Secuencia (2).csv"

H_PANEL = 860
W_FRAME = 1200
W_MALLA = 720
MARGEN = 44
WIN = "match de tiros — video (izq)  vs  malla del CSV (der)"

# correspondencias ya validadas para este clip (debug/homografia.py), solo --seed
SEED = {"313": (2411, 1217), "502": (1664, 1269),
        "801": (1687, 1014), "716": (2391, 978)}


# ----------------------------------------------------------------------
# DATOS Y AJUSTE (identico a debug/homografia.py)
# ----------------------------------------------------------------------
def load_holes(path):
    """Lee el CSV de secuencia -> {label: (X, Y)} y {label: t_detonacion_ms}."""
    holes, times = {}, {}
    for r in list(csv.reader(open(path)))[1:]:
        if len(r) >= 5 and r[4].strip().replace('.', '').isdigit():
            holes[r[0].strip()] = (float(r[1]), float(r[2]))
            times[r[0].strip()] = float(r[4])
    return holes, times


def affine_from_correspondences(world, img):
    """Replica getSafeAffineMatrix: LSQ afin con el mundo centrado (SVD).

    Se centra el mundo porque las coordenadas son UTM (~1e5): sin centrar, el
    sistema queda mal condicionado y la traslacion se come la precision.
    """
    world = np.asarray(world, float)
    img = np.asarray(img, float)
    c = world.mean(0)
    A = np.column_stack([world - c, np.ones(len(world))])
    X, *_ = np.linalg.lstsq(A, img, rcond=None)
    a, b, t1l = X[0, 0], X[1, 0], X[2, 0]
    cc, d, t2l = X[0, 1], X[1, 1], X[2, 1]
    return np.array([[a, b, t1l - a * c[0] - b * c[1]],
                     [cc, d, t2l - cc * c[0] - d * c[1]],
                     [0, 0, 1.0]])


def apply_H(H, pts):
    p = np.column_stack([np.asarray(pts, float), np.ones(len(pts))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def anclas_sugeridas(holes):
    """Los 4 tiros mas extremos (esquinas del hull rotado 45 grados).

    Maximizar la separacion entre anclas es lo que estabiliza el ajuste.
    """
    ks = list(holes)
    P = np.array([holes[k] for k in ks])
    P = P - P.mean(0)
    s, d = P[:, 0] + P[:, 1], P[:, 0] - P[:, 1]
    idx = {int(s.argmin()), int(s.argmax()), int(d.argmin()), int(d.argmax())}
    return [ks[i] for i in idx]


def wheel_delta(flags):
    """Delta de la rueda. cv2.getMouseWheelDelta no existe en todos los builds
    (falta en varios wheels de Windows), asi que se extrae a mano: viene en los
    16 bits altos de flags, como entero de 16 bits CON SIGNO."""
    try:
        return cv2.getMouseWheelDelta(flags)
    except AttributeError:
        d = (flags >> 16) & 0xFFFF
        return d - 0x10000 if d > 0x7FFF else d


def blit(dst, src, dx, dy):
    """Pega src en dst en (dx,dy) recortando lo que se sale del borde."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(W, dx + w), min(H, dy + h)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]


# ----------------------------------------------------------------------
# APLICACION
# ----------------------------------------------------------------------
class App:
    def __init__(self, clip, csv_path, seed=False):
        self.holes, self.times = load_holes(csv_path)
        self.keys = list(self.holes)
        self.anclas = anclas_sugeridas(self.holes)

        self.cap = cv2.VideoCapture(str(clip))
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fcache = {}

        # fondo promedio del pre-roll: mas limpio que cualquier frame suelto
        self.bg = None
        f = CACHE / "fondo.npz"
        if f.exists():
            m = np.load(f)["bg_mean"]
            self.bg = cv2.cvtColor(np.clip(m, 0, 255).astype(np.uint8),
                                   cv2.COLOR_GRAY2BGR)

        self.frame_idx = max(0, START - 10)      # antes de la detonacion
        self.use_bg = False
        self.img = None
        self.corr = dict(SEED) if seed else {}
        self.orden = list(self.corr)
        self.sel = None
        self.show_malla = True
        self.drag = None
        self.view = [0, 0, 1, 1]
        self.mouse = (W_FRAME // 2, H_PANEL // 2)   # para que +/- centren ahi
        self.zoom_tb = 100                          # ultimo valor del trackbar
        self._malla_map()
        self.load_image()
        self.reset_view()
        if self.corr:                    # con --seed: parte encuadrado
            self.encuadrar_malla()

    # ---------------- imagen base ----------------
    def load_image(self):
        if self.use_bg and self.bg is not None:
            self.img = self.bg
            return
        i = int(np.clip(self.frame_idx, 0, self.nframes - 1))
        if i not in self.fcache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = self.cap.read()
            if not ok:
                fr = np.zeros((self.fh, self.fw, 3), np.uint8)
            if len(self.fcache) > 12:            # el 4K pesa: cache acotado
                self.fcache.pop(next(iter(self.fcache)))
            self.fcache[i] = fr
        self.img = self.fcache[i]

    # ---------------- vista (zoom / pan) ----------------
    def fit_w(self):
        """Ancho de vista, en px del frame, que corresponde a zoom 1x."""
        ar = W_FRAME / H_PANEL
        return self.fw if self.fw / self.fh > ar else self.fh * ar

    def reset_view(self):
        vw = self.fit_w()
        vh = vw / (W_FRAME / H_PANEL)
        self.view = [(self.fw - vw) / 2, (self.fh - vh) / 2, vw, vh]

    def zoom_level(self):
        return self.fit_w() / max(self.view[2], 1e-6)

    def set_zoom(self, factor, anchor=None):
        """Fija el zoom absoluto (1.0 = cuadro completo) conservando el punto
        de anclaje; sin anclaje, conserva el centro de la vista."""
        factor = float(np.clip(factor, 1.0, 40.0))
        nw = self.fit_w() / factor
        nh = nw * H_PANEL / W_FRAME
        x, y, w, h = self.view
        if anchor is None:
            cx, cy = x + w / 2, y + h / 2
            self.view = [cx - nw / 2, cy - nh / 2, nw, nh]
        else:
            fx, fy = anchor
            rx, ry = (fx - x) / w, (fy - y) / h
            self.view = [fx - rx * nw, fy - ry * nh, nw, nh]

    def encuadrar_malla(self):
        """Lleva la vista al bounding box de los tiros proyectados.

        Es el atajo que evita buscar la tronadura a mano dentro del 4K.
        """
        H, _ = self.solve()
        if H is None:
            return False
        p = apply_H(H, [self.holes[k] for k in self.keys])
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        m = 0.30 * max(x1 - x0, y1 - y0)             # aire alrededor
        vw = max((x1 - x0) + 2 * m, ((y1 - y0) + 2 * m) * W_FRAME / H_PANEL)
        vh = vw * H_PANEL / W_FRAME
        self.view = [(x0 + x1) / 2 - vw / 2, (y0 + y1) / 2 - vh / 2, vw, vh]
        return True

    def f2v(self, px, py):
        x, y, w, _ = self.view
        s = W_FRAME / w
        return int(round((px - x) * s)), int(round((py - y) * s))

    def v2f(self, vx, vy):
        x, y, w, _ = self.view
        s = W_FRAME / w
        return x + vx / s, y + vy / s

    def zoom_at(self, vx, vy, factor):
        fx, fy = self.v2f(vx, vy)
        x, y, w, h = self.view
        # mismo rango que set_zoom: nunca mas lejos que el cuadro completo
        nw = float(np.clip(w / factor, self.fit_w() / 40.0, self.fit_w()))
        nh = nw * H_PANEL / W_FRAME
        rx, ry = (fx - x) / w, (fy - y) / h
        self.view = [fx - rx * nw, fy - ry * nh, nw, nh]

    # ---------------- malla (panel derecho) ----------------
    def _malla_map(self):
        P = np.array([self.holes[k] for k in self.keys])
        self.wmin = P.min(0)
        rng = np.maximum(P.max(0) - self.wmin, 1e-6)
        w, h = W_MALLA - 2 * MARGEN, H_PANEL - 2 * MARGEN
        self.ms = min(w / rng[0], h / rng[1])
        self.mo = (MARGEN + (w - rng[0] * self.ms) / 2,
                   MARGEN + (h - rng[1] * self.ms) / 2)

    def w2c(self, x, y):
        """Mundo -> canvas de la malla. Y se invierte: en el mundo crece hacia
        el norte, en la imagen hacia abajo."""
        cx = self.mo[0] + (x - self.wmin[0]) * self.ms
        cy = H_PANEL - (self.mo[1] + (y - self.wmin[1]) * self.ms)
        return int(round(cx)), int(round(cy))

    def tiro_cercano_malla(self, cx, cy, tol=26):
        best, bd = None, tol
        for k in self.keys:
            x, y = self.w2c(*self.holes[k])
            d = np.hypot(x - cx, y - cy)
            if d < bd:
                best, bd = k, d
        return best

    def tiro_cercano_proy(self, fx, fy, H, tol_px=90):
        """Sobre el frame: que tiro proyectado esta mas cerca del click."""
        proj = apply_H(H, [self.holes[k] for k in self.keys])
        d = np.hypot(proj[:, 0] - fx, proj[:, 1] - fy)
        i = int(d.argmin())
        _, _, w, _ = self.view
        tol = tol_px * (w / self.fw)        # la tolerancia sigue al zoom
        return self.keys[i] if d[i] < max(tol, 12) else None

    # ---------------- ajuste ----------------
    def solve(self):
        if len(self.corr) < 3:
            return None, None
        ks = list(self.corr)
        H = affine_from_correspondences([self.holes[k] for k in ks],
                                        [self.corr[k] for k in ks])
        pred = apply_H(H, [self.holes[k] for k in ks])
        err = {k: float(np.hypot(*(pred[i] - np.array(self.corr[k]))))
               for i, k in enumerate(ks)}
        return H, err

    def color_t(self, k):
        t = np.array([self.times[q] for q in self.keys])
        v = (self.times[k] - t.min()) / (np.ptp(t) + 1e-6)
        c = cv2.applyColorMap(np.uint8([[v * 255]]), cv2.COLORMAP_JET)[0, 0]
        return int(c[0]), int(c[1]), int(c[2])

    # ---------------- render ----------------
    def panel_frame(self, H, err):
        x, y, w, _ = self.view
        s = W_FRAME / w
        dst = np.full((H_PANEL, W_FRAME, 3), 28, np.uint8)
        sx0, sy0 = max(0, int(x)), max(0, int(y))
        sx1 = min(self.fw, int(np.ceil(x + w)))
        sy1 = min(self.fh, int(np.ceil(y + self.view[3])))
        if sx1 > sx0 and sy1 > sy0:
            sub = self.img[sy0:sy1, sx0:sx1]
            nw, nh = int(round((sx1 - sx0) * s)), int(round((sy1 - sy0) * s))
            if nw > 0 and nh > 0:
                interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
                sub = cv2.resize(sub, (nw, nh), interpolation=interp)
                blit(dst, sub, int(round((sx0 - x) * s)),
                     int(round((sy0 - y) * s)))

        if H is not None and self.show_malla:
            proj = apply_H(H, [self.holes[k] for k in self.keys])
            for k, (px, py) in zip(self.keys, proj):
                vx, vy = self.f2v(px, py)
                if -20 < vx < W_FRAME + 20 and -20 < vy < H_PANEL + 20:
                    cv2.circle(dst, (vx, vy), 5, self.color_t(k), -1)
                    cv2.circle(dst, (vx, vy), 5, (0, 0, 0), 1)

        for k, (px, py) in self.corr.items():
            vx, vy = self.f2v(px, py)
            col = (60, 255, 255) if k == self.sel else (80, 255, 80)
            cv2.drawMarker(dst, (vx, vy), col, cv2.MARKER_CROSS, 26, 2)
            e = f" {err[k]:.0f}px" if err else ""
            cv2.putText(dst, k + e, (vx + 12, vy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(dst, k + e, (vx + 12, vy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            if H is not None:
                p = apply_H(H, [self.holes[k]])[0]
                cv2.line(dst, (vx, vy), self.f2v(p[0], p[1]), col, 1)

        src = ("FONDO promedio del pre-roll" if self.use_bg and self.bg is not None
               else f"frame {self.frame_idx}")
        rms = (np.sqrt(np.mean([v ** 2 for v in err.values()])) if err else None)
        label(dst, f"IMAGEN: {src}    ZOOM {self.zoom_level():.1f}x",
              (14, 30), 0.68, (60, 255, 255))
        txt = (f"{len(self.corr)} correspondencias   RMS {rms:.1f} px"
               if rms is not None else
               f"{len(self.corr)} correspondencias   faltan "
               f"{3 - len(self.corr)} para calcular H")
        label(dst, txt, (14, 60), 0.6)
        label(dst, (f"ARMADO: tiro {self.sel} -> click en su pozo"
                    if self.sel else
                    ("click en un pozo: se asigna solo" if H is not None
                     else "elige un tiro en la malla (derecha)")),
              (14, H_PANEL - 18), 0.6,
              (60, 255, 255) if self.sel else (200, 200, 200))
        label(dst, "ZOOM: trackbar 'zoom %'  |  + -  sobre el cursor  |  "
                   "click der  |  E encuadra malla  |  R cuadro completo",
              (14, H_PANEL - 44), 0.48, (170, 210, 255))
        return dst

    def panel_malla(self, err):
        dst = np.full((H_PANEL, W_MALLA, 3), 22, np.uint8)
        for k in self.keys:
            cx, cy = self.w2c(*self.holes[k])
            if k in self.anclas and k not in self.corr:
                cv2.circle(dst, (cx, cy), 11, (230, 230, 230), 1)
            cv2.circle(dst, (cx, cy), 5, self.color_t(k), -1)
            if k in self.corr:
                cv2.circle(dst, (cx, cy), 9, (80, 255, 80), 2)
            if k == self.sel:
                cv2.circle(dst, (cx, cy), 14, (60, 255, 255), 2)
            cv2.putText(dst, k, (cx + 7, cy - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (170, 170, 170), 1, cv2.LINE_AA)
        label(dst, "MALLA DEL CSV", (12, 26), 0.62, (60, 255, 255))
        label(dst, "color = orden de detonacion", (12, 50), 0.45)
        label(dst, "circulo blanco = ancla sugerida", (12, 72), 0.45)
        label(dst, f"{len(self.keys)} tiros", (12, H_PANEL - 16), 0.5)
        return dst

    def render(self):
        H, err = self.solve()
        left = self.panel_frame(H, err)
        right = self.panel_malla(err)
        sep = np.full((H_PANEL, 5, 3), 200, np.uint8)
        return np.hstack([left, sep, right]), H, err

    # ---------------- mouse ----------------
    def on_mouse(self, ev, x, y, flags, _):
        en_malla = x > W_FRAME + 5
        if not en_malla:
            self.mouse = (x, y)          # para que +/- acerquen donde miras
        if ev == cv2.EVENT_MOUSEWHEEL and not en_malla:
            f = 1.25 if wheel_delta(flags) > 0 else 1 / 1.25
            self.zoom_at(x, y, f)
        elif ev == cv2.EVENT_RBUTTONDOWN and not en_malla:
            # respaldo de la rueda: en Windows no siempre llega al callback
            out = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            self.zoom_at(x, y, 1 / 1.5 if out else 1.5)
        elif ev == cv2.EVENT_MBUTTONDOWN:
            self.drag = (x, y, self.view[0], self.view[1])
        elif ev == cv2.EVENT_MBUTTONUP:
            self.drag = None
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            ox, oy, vx0, vy0 = self.drag
            s = W_FRAME / self.view[2]
            self.view[0] = vx0 - (x - ox) / s
            self.view[1] = vy0 - (y - oy) / s
        elif ev == cv2.EVENT_LBUTTONDOWN:
            if en_malla:
                k = self.tiro_cercano_malla(x - W_FRAME - 5, y)
                if k:
                    self.sel = None if k == self.sel else k
            else:
                fx, fy = self.v2f(x, y)
                k = self.sel
                if k is None:                       # sin armar: auto-asignar
                    H, _ = self.solve()
                    if H is not None:
                        k = self.tiro_cercano_proy(fx, fy, H)
                if k:
                    if k not in self.corr:
                        self.orden.append(k)
                    self.corr[k] = (int(round(fx)), int(round(fy)))
                    self.sel = None

    # ---------------- export ----------------
    def export(self):
        H, err = self.solve()
        if H is None:
            print("  faltan correspondencias (minimo 3)")
            return
        D_TIROS.mkdir(parents=True, exist_ok=True)
        rms = float(np.sqrt(np.mean([v ** 2 for v in err.values()])))
        data = {
            "h_matrix": [[float(v) for v in row] for row in H],
            "convencion": "metros -> pixeles; el pipeline usa su inversa",
            "correspondencias": {k: list(v) for k, v in self.corr.items()},
            "error_px_por_punto": {k: round(v, 2) for k, v in err.items()},
            "rms_px": round(rms, 2),
            "imagen_base": ("fondo_promedio_preroll"
                            if self.use_bg and self.bg is not None
                            else f"frame_{self.frame_idx}"),
            "csv": str(CSV_DEF.name),
        }
        (D_TIROS / "h_matrix.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {D_TIROS / 'h_matrix.json'}   RMS {rms:.2f} px")
        print("  h_matrix = [" + ", ".join(
            "[" + ", ".join(f"{v:.8g}" for v in row) + "]" for row in H) + "]")

        # validacion visual a resolucion COMPLETA
        vis = self.img.copy()
        proj = apply_H(H, [self.holes[k] for k in self.keys])
        for k, (px, py) in zip(self.keys, proj):
            cv2.circle(vis, (int(px), int(py)), 9, self.color_t(k), -1)
        for k, (px, py) in self.corr.items():
            cv2.drawMarker(vis, (px, py), (255, 255, 255),
                           cv2.MARKER_CROSS, 44, 3)
            cv2.putText(vis, f"{k} ({err[k]:.0f}px)", (px + 14, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        save(D_TIROS / "validacion_homografia.png",
             label(vis, f"H desde {len(self.corr)} tiros   RMS {rms:.2f} px"))


def main(seed=False, test=False, clip=CLIP, csv_path=CSV_DEF):
    app = App(clip, csv_path, seed=seed)
    if test:
        D_TIROS.mkdir(parents=True, exist_ok=True)
        img, H, err = app.render()
        save(D_TIROS / "match_preview.png", img)
        if H is not None:
            print("  H:\n" + np.array2string(H, precision=6, suppress_small=True))
            print("  error por punto:", {k: round(v, 2) for k, v in err.items()})
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W_FRAME + W_MALLA + 5, H_PANEL + 60)
    cv2.setMouseCallback(WIN, app.on_mouse)
    cv2.createTrackbar("frame", WIN, app.frame_idx, max(app.nframes - 1, 1),
                       lambda v: None)
    # zoom como trackbar: es la unica via que no depende del backend de mouse
    cv2.createTrackbar("zoom %", WIN, 100, 2000, lambda v: None)
    app.zoom_tb = int(round(app.zoom_level() * 100))
    cv2.setTrackbarPos("zoom %", WIN, max(100, min(2000, app.zoom_tb)))

    def sync_zoom():
        """Refleja en el trackbar el zoom que se cambio por otra via."""
        z = max(100, min(2000, int(round(app.zoom_level() * 100))))
        app.zoom_tb = z
        cv2.setTrackbarPos("zoom %", WIN, z)

    print(__doc__)
    while True:
        v = cv2.getTrackbarPos("frame", WIN)
        if v != app.frame_idx and not app.use_bg:
            app.frame_idx = v
            app.load_image()
        z = cv2.getTrackbarPos("zoom %", WIN)
        if z != app.zoom_tb:                  # lo movio el usuario
            app.zoom_tb = z
            app.set_zoom(max(z, 100) / 100.0)

        img, _, _ = app.render()
        cv2.imshow(WIN, img)
        # waitKeyEx (no waitKey): las flechas son codigos extendidos y el
        # "& 0xFF" habitual las borra, ademas de chocar con Q/R/S/T mayusculas
        kk = cv2.waitKeyEx(20)
        k = kk & 0xFF if kk > 0 else 255
        if k in (ord("q"), 27):
            break
        elif k == ord("g"):
            app.export()
        elif k == ord("r"):
            app.reset_view()
            sync_zoom()
        elif k == ord("e"):
            if app.encuadrar_malla():
                sync_zoom()
            else:
                print("  aun no hay H: marca 3 tiros primero")
        elif k == ord("m"):
            app.show_malla = not app.show_malla
        elif k == ord("f"):
            if app.bg is None:
                print("  no hay fondo: corre antes debug/pre_fondo.py")
            else:
                app.use_bg = not app.use_bg
                app.load_image()
        elif k == ord("u") and app.orden:
            app.corr.pop(app.orden.pop(), None)
        elif k == ord("d") and app.sel in app.corr:
            app.corr.pop(app.sel)
            app.orden = [q for q in app.orden if q != app.sel]
        elif k in (ord("+"), ord("=")):
            app.zoom_at(*app.mouse, 1.4)      # acerca donde esta el cursor
            sync_zoom()
        elif k in (ord("-"), ord("_")):
            app.zoom_at(*app.mouse, 1 / 1.4)
            sync_zoom()
        elif kk in (2424832, 65361):          # flecha izquierda (Win, Linux)
            app.view[0] -= app.view[2] * 0.15
        elif kk in (2555904, 65363):          # flecha derecha
            app.view[0] += app.view[2] * 0.15
        elif kk in (2490368, 65362):          # flecha arriba
            app.view[1] -= app.view[3] * 0.15
        elif kk in (2621440, 65364):          # flecha abajo
            app.view[1] += app.view[3] * 0.15
    app.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true",
                    help="parte con las 4 correspondencias ya validadas")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--video", default=str(CLIP))
    ap.add_argument("--csv", default=str(CSV_DEF))
    a = ap.parse_args()
    main(a.seed, a.test, Path(a.video), Path(a.csv))
