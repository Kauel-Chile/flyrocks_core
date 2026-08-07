"""
PASO 9 — VISOR DE CAPAS: superponer todo y ver si las capas se corresponden.

Responde la pregunta concreta: las binarias de intensidad y de linealidad se
solapan solo un 10%. Ese 10%, ¿son las trayectorias reales y el resto ruido?
El modo SOLAPE (tecla S) lo pinta para que se juzgue mirando, y el panel de
estadisticas lo cuantiza cruzando con la mascara pintada.

CAPAS DISPONIBLES
  binaria de LINEALIDAD    verde
  binaria de INTENSIDAD    magenta
  binaria de Z-SCORE       cian
  mascara PINTADA          rojo / verde / azul translucido
  MALLA de tiros           circulos amarillos (necesita 05_tiros/h_matrix.json)

MODO SOLAPE (tecla S) — compara linealidad vs intensidad:
  verde     solo la linealidad lo ve
  magenta   solo la intensidad lo ve
  BLANCO    lo ven las DOS      <- la hipotesis: aqui estan las trayectorias

CONTROLES
  1 2 3     prende/apaga binaria de linealidad / intensidad / z-score
  4         prende/apaga la mascara pintada
  5         prende/apaga la malla de tiros
  S         modo SOLAPE (linealidad vs intensidad)
  C         modo COMBINADO: conserva los trazos de linealidad que tienen encima
            un punto fuerte de intensidad, y les resta el verde de la pintada.
            La exigencia de evidencia se mueve con el umbral ALTO de INTENSIDAD.
  V         en modo combinado, quita o no la zona verde de la pintada
  B         cambia la imagen de fondo (linealidad / intensidad / z-score / negro)
  trackbars 1.AJUSTAR elige a que fuente le mueves umbrales; cada fuente
            recuerda los suyos. 2.ALTO y 3.BAJO son percentiles.
  ZOOM      trackbar "zoom %", teclas + / -, click derecho (Ctrl=alejar), rueda
  E         encuadra la malla de tiros      R  cuadro completo
  flechas / boton medio      desplazar
  G         exporta cada binaria por separado + union + solape + composicion
  Q / ESC   salir

    uv run python debug/pre_capas.py
    uv run python debug/pre_capas.py --test
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pre_common import CACHE, OUT, ROOT, ensure_dirs, to8, label, save
from pre_umbrales import hysteresis_guarded, length_filter, ncomp
import pre_pasada

D_CAPAS = OUT / "06_capas"
PINTADA = ROOT / "debug" / "out" / "6_mascaras" / "5_pintada.png"
H_JSON = OUT / "05_tiros" / "h_matrix.json"
CSV = ROOT / "debug" / "Secuencia (2).csv"

W_VIEW, H_VIEW = 1500, 830
WIN = "capas superpuestas"

COL = {"lin": (60, 255, 60), "int": (255, 60, 255), "z": (255, 255, 60)}
NOMBRE = {"lin": "LINEALIDAD", "int": "INTENSIDAD", "z": "Z-SCORE"}


def fusion(m):
    """(n_componentes, % de la mascara que esta en el componente MAYOR).

    Es el indicador de derrame: cuando un solo componente concentra la mayor
    parte del area, la mascara dejo de separar trayectorias y se fusiono en un
    unico blob. Ahi ya no aporta informacion aguas abajo, por mucha area que
    tenga. Por debajo de ~30% la mascara discrimina.
    """
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1 or m.sum() == 0:
        return 0, 0.0
    a = st[1:, cv2.CC_STAT_AREA]
    return n - 1, 100.0 * a.max() / m.sum()


_K3 = np.ones((3, 3), np.uint8)
MAX_COMP_TRAZO = 2500        # sobre esto el filtro de trazo se salta (costo)


def trazo_filter(mask, max_ext, stats_cc=None):
    """Descarta las 'pelusas': manchas hechas de muchas rayitas cortas cruzadas.

    POR QUE NO SIRVE EL FILTRO DE LARGO NI LA ELONGACION
      - largo_min mide la DIAGONAL DEL BOUNDING BOX: una pelusa de 100x100 px
        mide 141 y pasa cualquier umbral razonable, aunque este hecha de
        rayitas de 20 px.
      - la elongacion (minAreaRect) castiga las CURVAS: una parabola cerrada
        tiene el rectangulo envolvente casi cuadrado, y las parabolas son justo
        lo que se quiere conservar. Medido: las mataba.

    LO QUE SI DISTINGUE es la topologia: un trazo es un camino con 2 puntas;
    una pelusa es un manojo de rayitas con muchas puntas. Se esqueletiza el
    componente y se cuentan los EXTREMOS (pixeles del esqueleto con un solo
    vecino), normalizados por el largo del esqueleto — sin normalizar, las
    'barbas' que el adelgazamiento genera en bordes irregulares dominan el
    conteo y vuelven a matar las parabolas (verificado).

    max_ext = extremos permitidos por cada 100 px de esqueleto. 2 es un buen
    punto de partida; 0 desactiva el filtro.
    """
    if max_ext <= 0 or not mask.any():
        return mask
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1 or n - 1 > MAX_COMP_TRAZO:
        return mask
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 10:
            continue
        sub = ((lab[y:y + h, x:x + w] == i).astype(np.uint8)) * 255
        sk = cv2.ximgproc.thinning(
            sub, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
        L = int(sk.sum())
        if L < 20:
            continue
        grado = cv2.filter2D(sk.astype(np.uint8), cv2.CV_8U, _K3)[sk] - 1
        keep[i] = (100.0 * int((grado == 1).sum()) / L) <= max_ext
    keep[0] = False
    return keep[lab]


def blit(dst, src, dx, dy):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(W, dx + w), min(H, dy + h)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = src[y0 - dy:y1 - dy, x0 - dx:x1 - dx]


def wheel_delta(flags):
    try:
        return cv2.getMouseWheelDelta(flags)
    except AttributeError:                      # falta en varios builds Windows
        d = (flags >> 16) & 0xFFFF
        return d - 0x10000 if d > 0x7FFF else d


class Capas:
    def __init__(self):
        d = pre_pasada.load()
        self.src = {
            "lin": np.load(CACHE / "linealidad.npz")["lin_ff"].astype(np.float32),
            "int": d["acc_ff"].astype(np.float32),
            "z": d["acc_zff"].astype(np.float32),
        }
        self.fh, self.fw = self.src["int"].shape
        self.par = {k: dict(lo=950, hi=990, largo=80, trazo=0) for k in self.src}
        self.cache = {}
        self.scache = {}
        # Tabla de percentiles precalculada por fuente: np.percentile sobre 8.3M
        # floats cuesta ~0.1 s, y se llamaba dos veces por cada movimiento del
        # slider. Con la tabla, el umbral sale por interpolacion (instantaneo).
        self.qgrid = np.arange(0, 100.0001, 0.05)
        self.qtab = {k: np.percentile(v, self.qgrid).astype(np.float64)
                     for k, v in self.src.items()}
        self.on = {"lin": True, "int": True, "z": False,
                   "pint": False, "malla": False, "humo": False}
        self.solape = False
        self.combi = False
        self.quitar_verde = True
        self.ccache = {}
        # --- mascara de HUMO (filtro negativo automatico) ---
        # Idea del usuario: subir el largo minimo hasta que lo UNICO que queda
        # conectado sea la masa central. Las trayectorias son puntos sueltos o
        # trazos cortos y mueren; el humo, que es una masa continua, sobrevive.
        # Lo que queda no es señal: es exactamente lo que hay que restar.
        # OJO: el humo necesita umbrales PERMISIVOS (para que la masa quede
        # conectada), justo lo contrario que las semillas de intensidad, que
        # deben ser restrictivas. Por eso lleva percentil PROPIO y no reusa
        # self.binaria(): con el umbral restrictivo la masa ni se forma.
        self.humo_src = "int"
        self.humo_pct = 900            # percentil x10 -> p90
        self.humo_largo = 558          # el valor que encontro mirando
        self.quitar_humo = False
        self.humo_dilata = 0
        self.fondo = "lin"
        self.ajuste = "lin"

        # --- mascara pintada (opcional) ---
        self.pm = None
        if PINTADA.exists():
            im = cv2.imread(str(PINTADA))
            if im.shape[:2] != (self.fh, self.fw):
                im = cv2.resize(im, (self.fw, self.fh),
                                interpolation=cv2.INTER_NEAREST)
            b, g, r = (im[:, :, i].astype(int) for i in range(3))
            self.pm = {
                "rojo": (r > 100) & (r > g + 50) & (r > b + 50),
                "verde": (g > 100) & (g > r + 50) & (g > b + 50),
                "azul": (b > 100) & (b > r + 50) & (b > g + 50),
            }

        # --- malla de tiros (opcional) ---
        self.proj = None
        if H_JSON.exists() and CSV.exists():
            import csv as _csv
            H = np.array(json.loads(H_JSON.read_text(encoding="utf-8"))["h_matrix"])
            pts = []
            for r in list(_csv.reader(open(CSV)))[1:]:
                if len(r) >= 5 and r[4].strip().replace('.', '').isdigit():
                    pts.append((float(r[1]), float(r[2])))
            p = np.column_stack([np.array(pts), np.ones(len(pts))])
            q = (H @ p.T).T
            self.proj = q[:, :2] / q[:, 2:3]

        self.mouse = (W_VIEW // 2, H_VIEW // 2)
        self.drag = None
        self.zoom_tb = 100
        self.reset_view()

    def thr(self, k, p):
        """Umbral absoluto del percentil p, por interpolacion en la tabla."""
        return float(np.interp(p, self.qgrid, self.qtab[k]))

    # ---------------- binarias (con cache; recalcular en 4K cuesta) ----------
    def binaria(self, k):
        p = self.par[k]
        key = (k,) + tuple(sorted(p.items()))
        if key in self.cache:
            return self.cache[key]
        plo = min(p["lo"] / 10.0, 99.8)
        phi = min(max(p["hi"] / 10.0, plo + 0.1), 99.95)
        m = hysteresis_guarded(self.src[k], self.thr(k, plo),
                               self.thr(k, phi), 4000)
        m = length_filter(m, p["largo"])
        m = trazo_filter(m, p.get("trazo", 0) / 10.0)
        if len(self.cache) > 24:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = m
        return m

    # ---------------- vista ----------------
    def fit_w(self):
        ar = W_VIEW / H_VIEW
        return self.fw if self.fw / self.fh > ar else self.fh * ar

    def reset_view(self):
        vw = self.fit_w()
        self.view = [(self.fw - vw) / 2, (self.fh - vw / (W_VIEW / H_VIEW)) / 2,
                     vw, vw / (W_VIEW / H_VIEW)]

    def zoom_level(self):
        return self.fit_w() / max(self.view[2], 1e-6)

    def set_zoom(self, factor, anchor=None):
        factor = float(np.clip(factor, 1.0, 40.0))
        nw = self.fit_w() / factor
        nh = nw * H_VIEW / W_VIEW
        x, y, w, h = self.view
        fx, fy = anchor if anchor else (x + w / 2, y + h / 2)
        rx, ry = ((fx - x) / w, (fy - y) / h) if anchor else (0.5, 0.5)
        self.view = [fx - rx * nw, fy - ry * nh, nw, nh]

    def zoom_at(self, vx, vy, f):
        x, y, w, h = self.view
        s = W_VIEW / w
        self.set_zoom(self.zoom_level() * f, (x + vx / s, y + vy / s))

    def encuadrar(self):
        if self.proj is None:
            return False
        x0, y0 = self.proj.min(0)
        x1, y1 = self.proj.max(0)
        m = 0.30 * max(x1 - x0, y1 - y0)
        vw = max((x1 - x0) + 2 * m, ((y1 - y0) + 2 * m) * W_VIEW / H_VIEW)
        vh = vw * H_VIEW / W_VIEW
        self.view = [(x0 + x1) / 2 - vw / 2, (y0 + y1) / 2 - vh / 2, vw, vh]
        return True

    def vista(self, arr, nearest=False):
        """Recorta la region visible de una capa full-res y la escala al panel."""
        x, y, w, h = self.view
        s = W_VIEW / w
        out = np.zeros((H_VIEW, W_VIEW) + arr.shape[2:], arr.dtype)
        sx0, sy0 = max(0, int(x)), max(0, int(y))
        sx1, sy1 = min(self.fw, int(np.ceil(x + w))), min(self.fh, int(np.ceil(y + h)))
        if sx1 <= sx0 or sy1 <= sy0:
            return out
        sub = arr[sy0:sy1, sx0:sx1]
        nw, nh = int(round((sx1 - sx0) * s)), int(round((sy1 - sy0) * s))
        if nw <= 0 or nh <= 0:
            return out
        interp = (cv2.INTER_NEAREST if nearest or arr.dtype == bool
                  else (cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR))
        sub = cv2.resize(sub.astype(np.uint8) if arr.dtype == bool else sub,
                         (nw, nh), interpolation=interp)
        if arr.dtype == bool:
            sub = sub.astype(bool)
        blit(out, sub, int(round((sx0 - x) * s)), int(round((sy0 - y) * s)))
        return out

    def f2v(self, px, py):
        x, y, w, _ = self.view
        s = W_VIEW / w
        return int(round((px - x) * s)), int(round((py - y) * s))

    # ---------------- mascara de HUMO (filtro negativo) ---------------------
    def humo(self):
        """Lo que sobrevive a un largo minimo MUY alto: la masa central.

        Es el mismo length_filter, pero leido al reves. Con un umbral normal se
        usa para quedarse con los trazos largos; subiendolo hasta varios cientos
        de px, las trayectorias (puntos sueltos o trazos cortos) desaparecen y
        solo queda lo que esta conectado en masa, que es el humo. El resultado
        sirve como mascara NEGATIVA, derivada de la imagen y no dibujada a mano.
        """
        k = self.humo_src
        key = ("humo", k, self.humo_pct, self.humo_largo, self.humo_dilata)
        if key in self.ccache:
            return self.ccache[key]
        # umbral simple con percentil propio (no la binaria calibrada)
        h = self.src[k] > self.thr(k, min(self.humo_pct / 10.0, 99.9))
        h = length_filter(h, self.humo_largo)
        if self.humo_dilata > 0:
            r = int(self.humo_dilata) | 1
            h = cv2.dilate(h.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))) > 0
        if len(self.ccache) > 12:
            self.ccache.pop(next(iter(self.ccache)))
        self.ccache[key] = h
        return h

    # ---------------- combinacion: espacio validado por evidencia -----------
    def combinada(self):
        """Se queda con los trazos de LINEALIDAD que tienen encima al menos un
        punto fuerte de INTENSIDAD, y les resta el verde de la pintada.

        La idea: la linealidad dice "esto tiene forma de trayectoria", pero la
        textura del terreno y el humo peinado tambien tienen forma de linea. La
        intensidad es evidencia INDEPENDIENTE (hubo un cambio fuerte de brillo
        ahi). Un trazo respaldado por las dos cosas es mucho mas creible que uno
        que solo cumple la forma.

        Devuelve (conservado, descartado, puntos, n_total, n_conservados).
        """
        p_esp, p_sem = self.par["lin"], self.par["int"]
        key = ("comb",) + tuple(sorted(p_esp.items())) + \
              (p_sem["hi"], self.quitar_verde, self.quitar_humo, self.humo_src,
               self.humo_pct, self.humo_largo, self.humo_dilata)
        if key in self.ccache:
            return self.ccache[key]
        esp = self.binaria("lin")
        # los puntos son el umbral ALTO de intensidad: mover ese slider mueve
        # cuanta evidencia se exige
        pts = self.src["int"] > self.thr("int", min(p_sem["hi"] / 10.0, 99.95))
        n, lab, _, _ = cv2.connectedComponentsWithStats(esp.astype(np.uint8), 8)
        tiene = np.zeros(max(n, 1), bool)
        if n > 1:
            tiene[np.unique(lab[pts])] = True
            tiene[0] = False
        keep = tiene[lab]
        if self.quitar_verde and self.pm:
            keep = keep & ~self.pm["verde"]
        if self.quitar_humo:
            keep = keep & ~self.humo()
        out = (keep, esp & ~keep, pts, n - 1, int(tiene.sum()))
        if len(self.ccache) > 12:
            self.ccache.pop(next(iter(self.ccache)))
        self.ccache[key] = out
        return out

    # ---------------- estadisticas ----------------
    def stats(self):
        """Cruza cada region con la pintada. Es el test de la hipotesis:
        si el solape cae en ROJO/AZUL y lo exclusivo cae en VERDE, entonces
        el solape es señal y lo exclusivo es ruido."""
        skey = (tuple(self.par["lin"].values()), tuple(self.par["int"].values()))
        if skey in self.scache:          # si no, se recalcula en CADA frame
            return self.scache[skey]
        A, B = self.binaria("lin"), self.binaria("int")
        inter, solo_a, solo_b = A & B, A & ~B, B & ~A
        tot = self.fh * self.fw
        r = {"pct_lin": 100 * A.mean(), "pct_int": 100 * B.mean(),
             "pct_amb": 100 * inter.mean(),
             "jaccard": inter.sum() / max((A | B).sum(), 1),
             "n_lin": ncomp(A), "n_int": ncomp(B), "n_amb": ncomp(inter)}
        if self.pm:
            interes = self.pm["rojo"] | self.pm["azul"]
            for nom, m in (("amb", inter), ("solo_lin", solo_a), ("solo_int", solo_b)):
                n = max(m.sum(), 1)
                r[f"{nom}_interes"] = 100 * (m & interes).sum() / n
                r[f"{nom}_verde"] = 100 * (m & self.pm["verde"]).sum() / n
        if len(self.scache) > 40:
            self.scache.pop(next(iter(self.scache)))
        self.scache[skey] = r
        return r

    # ---------------- render ----------------
    def render(self):
        base = self.src.get(self.fondo)
        if base is None:
            img = np.zeros((H_VIEW, W_VIEW, 3), np.uint8)
        else:
            g = self.vista(to8(base, hi_pct=99.8))
            img = cv2.cvtColor((g * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if self.combi:
            keep, desc, pts, _, _ = self.combinada()
            # orden deliberado: la evidencia va DEBAJO. Es densa en la zona del
            # blast y si se dibuja encima tapa justo lo que hay que evaluar.
            img[self.vista(pts)] = (40, 150, 190)       # la evidencia, apagada
            img[self.vista(desc)] = (70, 70, 90)        # descartado
            img[self.vista(keep)] = (60, 255, 60)       # lo que sobrevive
        elif self.solape:
            A, B = self.vista(self.binaria("lin")), self.vista(self.binaria("int"))
            img[A & ~B] = (60, 200, 60)
            img[B & ~A] = (200, 60, 200)
            img[A & B] = (255, 255, 255)        # <- lo que ven las dos
        else:
            for k in ("lin", "int", "z"):
                if self.on[k]:
                    img[self.vista(self.binaria(k))] = COL[k]

        if self.on.get("humo"):
            hv = self.vista(self.humo())
            img[hv] = (180, 90, 200) if not self.combi else (110, 55, 125)

        if self.on["pint"] and self.pm:
            ov = img.copy()
            for nom, col in (("verde", (0, 200, 0)), ("rojo", (0, 0, 220)),
                             ("azul", (220, 0, 0))):
                ov[self.vista(self.pm[nom])] = col
            img = cv2.addWeighted(img, 0.72, ov, 0.28, 0)

        if self.on["malla"] and self.proj is not None:
            for px, py in self.proj:
                vx, vy = self.f2v(px, py)
                if 0 <= vx < W_VIEW and 0 <= vy < H_VIEW:
                    cv2.circle(img, (vx, vy), 4, (0, 230, 255), 1)

        return self.overlay_texto(img)

    def overlay_texto(self, img):
        st = self.stats()
        y = 30
        if self.combi:
            keep, desc, pts, ntot, nok = self.combinada()
            label(img, "MODO COMBINADO — trazos con evidencia de intensidad",
                  (14, y), 0.72, (60, 255, 60)); y += 30
            label(img, "verde = se conserva   gris = descartado (sin evidencia "
                       "o en zona verde)   celeste = la evidencia",
                  (14, y), 0.5); y += 26
            label(img, f"{nok} de {ntot} trazos tienen evidencia   "
                       f"({100*keep.mean():.2f}% del cuadro queda)",
                  (14, y), 0.58, (60, 255, 60)); y += 26
            if self.pm is not None:
                interes = self.pm["rojo"] | self.pm["azul"]
                base = 100 * interes.mean()
                en = 100 * interes[keep].mean() if keep.sum() else 0
                label(img, f"cae en zona de interes: {en:.1f}%   "
                           f"(al azar seria {base:.1f}%)   "
                           f"quitar verde: {'ON' if self.quitar_verde else 'off'} (V)"
                           f"   quitar humo: {'ON' if self.quitar_humo else 'off'} (H)",
                      (14, y), 0.52, (200, 220, 255)); y += 28
            if self.quitar_humo or self.on.get("humo"):
                hm = self.humo()
                label(img, f"MASCARA DE HUMO ({NOMBRE[self.humo_src]} "
                           f"p{self.humo_pct/10:.1f}, largo>{self.humo_largo}px): "
                           f"{100*hm.mean():.2f}% del cuadro   [N cambia fuente]",
                      (14, y), 0.5, (200, 130, 220)); y += 26
        elif self.solape:
            label(img, "MODO SOLAPE   blanco = lo ven LAS DOS", (14, y), 0.72,
                  (255, 255, 255)); y += 30
            label(img, f"verde solo linealidad ({st['pct_lin']:.2f}%)   "
                       f"magenta solo intensidad ({st['pct_int']:.2f}%)",
                  (14, y), 0.55); y += 26
            label(img, f"AMBAS {st['pct_amb']:.3f}% del cuadro   "
                       f"{st['n_amb']} componentes   Jaccard {st['jaccard']:.2f}",
                  (14, y), 0.58, (255, 255, 255)); y += 30
        else:
            act = [f"{NOMBRE[k]} {st['pct_' + k] if k != 'z' else 0:.2f}%"
                   for k in ("lin", "int", "z") if self.on[k]]
            label(img, "CAPAS: " + (", ".join(act) if act else "ninguna"),
                  (14, y), 0.62, (60, 255, 255)); y += 30

        if self.pm and "amb_interes" in st:
            label(img, "cruce con tu mascara pintada "
                       "(cuanto cae en ROJO/AZUL = zona de interes):",
                  (14, y), 0.5, (200, 220, 255)); y += 24
            for nom, txt in (("amb", "ven las DOS"), ("solo_lin", "solo linealidad"),
                             ("solo_int", "solo intensidad")):
                label(img, f"   {txt:<18} {st[nom + '_interes']:5.1f}% en interes   "
                           f"{st[nom + '_verde']:5.1f}% en verde (se descarta)",
                      (14, y), 0.5); y += 23

        p = self.par[self.ajuste]
        k = self.ajuste
        b = self.binaria(k)
        nc, fus = fusion(b)
        label(img, f"AJUSTANDO: {NOMBRE[k]}   "
                   f"hi=p{p['hi']/10:.1f} ({self.thr(k, p['hi']/10):.1f})  "
                   f"lo=p{p['lo']/10:.1f} ({self.thr(k, p['lo']/10):.1f})  "
                   f"largo>={p['largo']}px   ->  {100*b.mean():.2f}% del cuadro, "
                   f"{nc} comps", (14, H_VIEW - 88), 0.55, (60, 255, 255))
        # semaforo de derrame: es el numero que dice si el umbral se paso
        col = ((80, 255, 80) if fus < 30 else
               (60, 220, 255) if fus < 55 else (60, 60, 255))
        label(img, f"FUSION: {fus:.0f}% de la mascara esta en UN solo componente"
                   + ("   (ok, separa trayectorias)" if fus < 30 else
                      "   (ojo: se esta fusionando)" if fus < 55 else
                      "   <- DERRAME: es un blob, no separa nada"),
              (14, H_VIEW - 62), 0.55, col)
        label(img, f"fondo={NOMBRE.get(self.fondo, 'negro')}   "
                   f"zoom {self.zoom_level():.1f}x   "
                   f"1/2/3 binarias  4 pintada  5 malla  S solape  B fondo  G exporta",
              (14, H_VIEW - 34), 0.5)
        label(img, "pintada: " + ("ON" if self.on["pint"] else "off")
              + "   malla: " + ("ON" if self.on["malla"] else "off"),
              (14, H_VIEW - 10), 0.5)
        return img

    # ---------------- mouse ----------------
    def on_mouse(self, ev, x, y, flags, _):
        self.mouse = (x, y)
        if ev == cv2.EVENT_MOUSEWHEEL:
            self.zoom_at(x, y, 1.25 if wheel_delta(flags) > 0 else 1 / 1.25)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            out = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            self.zoom_at(x, y, 1 / 1.5 if out else 1.5)
        elif ev == cv2.EVENT_MBUTTONDOWN:
            self.drag = (x, y, self.view[0], self.view[1])
        elif ev == cv2.EVENT_MBUTTONUP:
            self.drag = None
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            ox, oy, vx0, vy0 = self.drag
            s = W_VIEW / self.view[2]
            self.view[0] = vx0 - (x - ox) / s
            self.view[1] = vy0 - (y - oy) / s

    # ---------------- export ----------------
    def export(self):
        D_CAPAS.mkdir(parents=True, exist_ok=True)
        st = self.stats()
        # Se exportan SIEMPRE las tres: los toggles 1/2/3 son de VISUALIZACION,
        # y apagar una capa para ver mejor no debe hacer que se pierda al
        # guardar. Los umbrales de las tres estan calibrados y guardados.
        union = None
        for k in ("lin", "int", "z"):
            m = self.binaria(k)
            save(D_CAPAS / f"binaria_{NOMBRE[k].lower()}.png",
                 (m * 255).astype(np.uint8))
            union = m if union is None else (union | m)
        save(D_CAPAS / "union.png", (union * 255).astype(np.uint8))
        A, B = self.binaria("lin"), self.binaria("int")
        sol = np.zeros((self.fh, self.fw, 3), np.uint8)
        sol[A & ~B] = (60, 200, 60)
        sol[B & ~A] = (200, 60, 200)
        sol[A & B] = (255, 255, 255)
        save(D_CAPAS / "solape_lin_vs_int.png", sol)
        keep, _, _, ntot, nok = self.combinada()
        save(D_CAPAS / "combinada.png", (keep * 255).astype(np.uint8))
        save(D_CAPAS / "composicion.png", self.render())
        params = {
            "capas_visibles_al_exportar": [NOMBRE[k] for k in ("lin", "int", "z")
                                           if self.on[k]],
            "nota_capas": "se exportan las tres siempre; 'visibles' es solo "
                          "que estaba dibujado en pantalla",
            "umbrales": {NOMBRE[k]: {"percentil_lo": self.par[k]["lo"] / 10,
                                     "percentil_hi": self.par[k]["hi"] / 10,
                                     "largo_min_px": self.par[k]["largo"]}
                         for k in self.src},
            "estadisticas": {k: (round(v, 3) if isinstance(v, float) else int(v))
                             for k, v in st.items()},
            "combinada": {"trazos_totales": ntot, "trazos_con_evidencia": nok,
                          "pct_cuadro": round(float(100 * keep.mean()), 3),
                          "verde_descartado": self.quitar_verde,
                          "regla": "trazos de linealidad que contienen al menos "
                                   "un pixel sobre el umbral ALTO de intensidad"},
            "nota": "capas SEPARADAS; la pintada no se fusiona con las binarias",
        }
        (D_CAPAS / "capas_params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {D_CAPAS / 'capas_params.json'}")


def main(test=False):
    ensure_dirs()
    c = Capas()
    if test:
        D_CAPAS.mkdir(parents=True, exist_ok=True)
        c.encuadrar()
        c.solape = True
        save(D_CAPAS / "preview_solape.png", c.render())
        c.solape = False
        c.on["pint"] = c.on["malla"] = True
        save(D_CAPAS / "preview_capas.png", c.render())
        st = c.stats()
        print("  " + json.dumps({k: round(v, 3) if isinstance(v, float) else int(v)
                                 for k, v in st.items()}, indent=2))
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W_VIEW, H_VIEW + 150)
    cv2.setMouseCallback(WIN, c.on_mouse)
    cv2.createTrackbar("1.AJUSTAR 0lin 1int 2zsc", WIN, 0, 2, lambda v: None)
    cv2.createTrackbar("2.ALTO  hi x10", WIN, c.par["lin"]["hi"], 999, lambda v: None)
    cv2.createTrackbar("3.BAJO  lo x10", WIN, c.par["lin"]["lo"], 998, lambda v: None)
    cv2.createTrackbar("4.largo_min px", WIN, c.par["lin"]["largo"], 600, lambda v: None)
    # 0 = sin filtro; 20 = 2.0 extremos por 100px de esqueleto (buen punto)
    cv2.createTrackbar("5.anti-pelusa x10", WIN, c.par["lin"]["trazo"], 60,
                       lambda v: None)
    cv2.createTrackbar("6.HUMO largo px", WIN, c.humo_largo, 1500, lambda v: None)
    cv2.createTrackbar("7.HUMO pctil x10", WIN, c.humo_pct, 999, lambda v: None)
    cv2.createTrackbar("8.zoom %", WIN, 100, 2000, lambda v: None)

    orden = ["lin", "int", "z"]
    last_aj = 0
    print(__doc__)
    while True:
        aj = cv2.getTrackbarPos("1.AJUSTAR 0lin 1int 2zsc", WIN)
        if aj != last_aj:
            # al cambiar de fuente hay que RECUPERAR sus umbrales en los
            # sliders, y saltarse la lectura de este ciclo: los trackbars aun
            # tienen los valores de la fuente anterior y se los copiaria.
            last_aj = aj
            c.ajuste = orden[aj]
            p = c.par[c.ajuste]
            cv2.setTrackbarPos("2.ALTO  hi x10", WIN, p["hi"])
            cv2.setTrackbarPos("3.BAJO  lo x10", WIN, p["lo"])
            cv2.setTrackbarPos("4.largo_min px", WIN, p["largo"])
            cv2.setTrackbarPos("5.anti-pelusa x10", WIN, p["trazo"])
        else:
            # Sin debounce: recalcular cuesta ~0.12 s y aplicarlo de inmediato
            # es lo unico que da respuesta al mover el slider.
            c.par[c.ajuste] = dict(
                hi=cv2.getTrackbarPos("2.ALTO  hi x10", WIN),
                lo=cv2.getTrackbarPos("3.BAJO  lo x10", WIN),
                largo=cv2.getTrackbarPos("4.largo_min px", WIN),
                trazo=cv2.getTrackbarPos("5.anti-pelusa x10", WIN))

        c.humo_largo = cv2.getTrackbarPos("6.HUMO largo px", WIN)
        c.humo_pct = cv2.getTrackbarPos("7.HUMO pctil x10", WIN)

        z = cv2.getTrackbarPos("8.zoom %", WIN)
        if z != c.zoom_tb:
            c.zoom_tb = z
            c.set_zoom(max(z, 100) / 100.0)

        cv2.imshow(WIN, c.render())
        kk = cv2.waitKeyEx(20)
        k = kk & 0xFF if kk > 0 else 255

        def sync():
            c.zoom_tb = max(100, min(2000, int(round(c.zoom_level() * 100))))
            cv2.setTrackbarPos("8.zoom %", WIN, c.zoom_tb)

        if k in (ord("q"), 27):
            break
        elif k in (ord("1"), ord("2"), ord("3")):
            key = orden[k - ord("1")]
            c.on[key] = not c.on[key]
        elif k == ord("4"):
            c.on["pint"] = not c.on["pint"]
        elif k == ord("5"):
            c.on["malla"] = not c.on["malla"]
        elif k == ord("s"):
            c.solape = not c.solape
            c.combi = False
        elif k == ord("c"):
            c.combi = not c.combi
            c.solape = False
        elif k == ord("v"):
            c.quitar_verde = not c.quitar_verde
        elif k == ord("h"):
            c.quitar_humo = not c.quitar_humo
        elif k == ord("6"):
            c.on["humo"] = not c.on["humo"]
        elif k == ord("n"):
            ciclo = ["int", "z", "lin"]
            c.humo_src = ciclo[(ciclo.index(c.humo_src) + 1) % 3]
        elif k == ord("b"):
            ciclo = ["lin", "int", "z", "negro"]
            c.fondo = ciclo[(ciclo.index(c.fondo) + 1) % 4]
        elif k == ord("e"):
            if c.encuadrar():
                sync()
        elif k == ord("r"):
            c.reset_view(); sync()
        elif k == ord("g"):
            c.export()
        elif k in (ord("+"), ord("=")):
            c.zoom_at(*c.mouse, 1.4); sync()
        elif k in (ord("-"), ord("_")):
            c.zoom_at(*c.mouse, 1 / 1.4); sync()
        elif kk in (2424832, 65361):
            c.view[0] -= c.view[2] * 0.15
        elif kk in (2555904, 65363):
            c.view[0] += c.view[2] * 0.15
        elif kk in (2490368, 65362):
            c.view[1] -= c.view[3] * 0.15
        elif kk in (2621440, 65364):
            c.view[1] += c.view[3] * 0.15
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args().test)
