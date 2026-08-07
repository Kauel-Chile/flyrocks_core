"""
Utilidades compartidas de la etapa de PREPROCESAMIENTO (ver debug/PREPROCESO.md).

NO modifica nada del pipeline existente. Solo lee el clip y produce mascaras
e imagenes en debug/out/7_preproceso/.

Piezas clave:
  - estimacion de movimiento global en escala reducida (rapido en 4K)
  - registro de CADA frame directo contra el FONDO (sin deriva acumulada)
  - normalizacion robusta de iluminacion (mediana + MAD), para sobrevivir al
    reajuste de auto-exposure que provoca el flash de la detonacion
  - helpers de visualizacion (normalizado por percentiles, etiquetas, mosaicos)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))

# ----------------------------------------------------------------------
# RUTAS
# ----------------------------------------------------------------------
SCRATCH = Path(
    r"C:\Users\carlo\AppData\Local\Temp\claude"
    r"\D--PROYECTOS-Enaex---Flyrocks-detovision-standalone-flyrocks-core"
    r"\3f0c6b88-01e1-4a43-9f40-09651754d37b\scratchpad"
)
CLIP = SCRATCH / "clip_full.mp4"          # 3840x2160, 453 frames, 29.97 fps
START = 48                                 # frame de detonacion
OUT = ROOT / "debug" / "out" / "7_preproceso"
CACHE = OUT / "_cache"

D_BASELINE = OUT / "00_baseline"
D_FONDO = OUT / "01_fondo"
D_ZSCORE = OUT / "02_zscore"
D_LINEAL = OUT / "03_linealidad"
D_BINARIA = OUT / "04_binaria"

NOISE_THRESHOLD = 8       # el umbral global fijo del pipeline actual (baseline)
# Piso de sigma: evita dividir por ~0 en zonas planas. El video es H.264, asi que
# hay bloques artificialmente lisos donde sigma->0; sin piso, el z-score explota.
# Medido en el pre-roll: diff_sigma mediana ~0.46, p95 ~1.0 (ruido MUY bajo, el
# blur 3x3 ya lo suaviza) => el umbral fijo de 8 equivale a ~17 sigmas.
SIGMA_FLOOR = 0.3


def ensure_dirs():
    for d in (OUT, CACHE, D_BASELINE, D_FONDO, D_ZSCORE, D_LINEAL, D_BINARIA):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# MOVIMIENTO GLOBAL (estimado en escala reducida: en 4K es ~4x mas rapido)
# ----------------------------------------------------------------------
MOTION_SCALE = 0.5


def estimate_motion(src_gray, dst_gray, scale=MOTION_SCALE):
    """Afin parcial que mapea coordenadas de src -> dst. None si no hay soporte.

    RANSAC descarta rocas/humo como outliers y se queda con el terreno.
    Se estima sobre imagenes reducidas y luego se re-escala la traslacion:
    para M_s estimada a escala s, la parte lineal 2x2 es identica y la
    traslacion se divide por s.
    """
    # LK exige uint8
    s_small = cv2.resize(np.clip(src_gray, 0, 255).astype(np.uint8), None,
                         fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    d_small = cv2.resize(np.clip(dst_gray, 0, 255).astype(np.uint8), None,
                         fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    pts = cv2.goodFeaturesToTrack(s_small, maxCorners=1200,
                                  qualityLevel=0.01, minDistance=12)
    if pts is None or len(pts) < 12:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(s_small, d_small, pts, None)
    if nxt is None:
        return None
    good = status.ravel() == 1
    if good.sum() < 12:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=3
    )
    if M is None or inliers is None or int(inliers.sum()) < 12:
        return None
    M = M.copy()
    M[:, 2] /= scale
    return M


def warp_to(img, M, shape):
    """Aplica el afin 2x3 M (o identidad si M is None)."""
    if M is None:
        return img
    h, w = shape
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compose(A, B):
    """Compone dos afines 2x3: devuelve el afin equivalente a aplicar B y luego A."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2]


def invert(M):
    return cv2.invertAffineTransform(M)


# ----------------------------------------------------------------------
# NORMALIZACION DE ILUMINACION (robusta al flash de la detonacion)
# ----------------------------------------------------------------------
def illum_stats(img, mask=None):
    """Mediana y MAD de la imagen (estadisticos robustos: sobreviven a que una
    parte grande del cuadro se tape con humo)."""
    v = img[mask] if mask is not None else img.ravel()
    v = v[::7]                      # submuestreo: en 4K basta y es mucho mas rapido
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) + 1e-6
    return med, mad


def match_illum(img, ref_med, ref_mad, mask=None):
    """Reescala img para que su mediana/MAD coincidan con las de la referencia.

    Sin esto, el reajuste de auto-exposure de la camara tras el flash aparece
    como un salto global de intensidad que el diff interpreta como "todo se movio".
    """
    med, mad = illum_stats(img, mask)
    gain = ref_mad / mad
    gain = float(np.clip(gain, 0.5, 2.0))      # sin correcciones absurdas
    return (img - med) * gain + ref_med


# ----------------------------------------------------------------------
# VISUALIZACION
# ----------------------------------------------------------------------
def to8(img, lo_pct=0.0, hi_pct=99.7, lo=None, hi=None):
    """Normaliza a uint8 por percentiles (robusto a outliers muy brillantes)."""
    f = img.astype(np.float32)
    if lo is None:
        lo = float(np.percentile(f, lo_pct))
    if hi is None:
        hi = float(np.percentile(f, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def falsecolor(img8, cmap=cv2.COLORMAP_TURBO):
    return cv2.applyColorMap(img8, cmap)


_ASCII = {"—": "-", "–": "-", "≥": ">=", "≤": "<=", "±": "+/-", "·": ".",
          "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n",
          "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ñ": "N",
          "σ": "sigma", "“": '"', "”": '"', "’": "'"}


def _ascii(text):
    """cv2.putText solo dibuja ASCII: lo demas sale como '???'."""
    return "".join(_ASCII.get(c, c if ord(c) < 128 else "?") for c in text)


def label(img, text, org=(30, 70), scale=None, color=(255, 255, 255)):
    """Escribe texto con reborde negro (legible sobre cualquier fondo)."""
    text = _ascii(text)
    out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if scale is None:
        scale = max(1.0, out.shape[1] / 1400.0)
    th = max(2, int(scale * 2))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), th + 4, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, th, cv2.LINE_AA)
    return out


def side_by_side(items, max_w=3400, gap=12):
    """items = [(img, titulo), ...] -> mosaico horizontal etiquetado."""
    tiles = []
    for img, title in items:
        t = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t = label(t.copy(), title)
        tiles.append(t)
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    total_w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((h, total_w, 3), 40, np.uint8)
    x = 0
    for t in tiles:
        canvas[:, x:x + t.shape[1]] = t
        x += t.shape[1] + gap
    if canvas.shape[1] > max_w:
        sc = max_w / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=sc, fy=sc,
                            interpolation=cv2.INTER_AREA)
    return canvas


def grid(items, cols, max_w=3400, gap=8):
    """items = [(img, titulo), ...] -> mosaico en grilla etiquetado."""
    tiles = []
    for img, title in items:
        t = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        tiles.append(label(t.copy(), title))
    h, w = tiles[0].shape[:2]
    tiles = [cv2.resize(t, (w, h)) for t in tiles]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * h + gap * (rows - 1),
                      cols * w + gap * (cols - 1), 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * (h + gap):r * (h + gap) + h,
               c * (w + gap):c * (w + gap) + w] = t
    if canvas.shape[1] > max_w:
        sc = max_w / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=sc, fy=sc,
                            interpolation=cv2.INTER_AREA)
    return canvas


def save(path, img):
    cv2.imwrite(str(path), img)
    print(f"  -> {path.relative_to(ROOT)}")
