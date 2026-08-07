"""
Camino de las MASCARAS — genera las capas complementarias en UNA sola pasada y
una vista combinada, para analizar si sumarlas aporta valor.

Capas (todas misma ventana/encuadre, en debug/out/6_mascaras/):
  1_intensidad.png       -> que tan fuerte fue el cambio (gris)
  2_temporal.png         -> cuando llego el movimiento (color, primera llegada)
  3_combinada.png        -> HSV: color=cuando, brillo=intensidad (las dos juntas)
  4_combinada_malla.png  -> la combinada + la malla de tiros (rojo)
  5_pintada.png          -> la mascara pintada por el usuario (referencia)

    uv run python debug/mask_combined.py
"""
import sys
import csv
import shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from harness import estimate_global_motion, NOISE_THRESHOLD          # noqa: E402

SCRATCH = Path(
    r"C:\Users\carlo\AppData\Local\Temp\claude"
    r"\D--PROYECTOS-Enaex---Flyrocks-detovision-standalone-flyrocks-core"
    r"\3f0c6b88-01e1-4a43-9f40-09651754d37b\scratchpad"
)
CLIP = SCRATCH / "clip_full.mp4"                 # ventana completa (12.5-27.6s)
START = 48                                        # frame de detonacion
OUT = ROOT / "debug" / "out" / "6_mascaras"
CACHE = ROOT / "debug" / "out" / "_cache"
PAINT = ROOT / "debug" / "out" / "4_fase0_referencias" / "lienzo_para_pintar_gris - Copy.png"


def one_pass(clip):
    """Devuelve acc (intensidad max por pixel) y first (frame de 1a llegada)."""
    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, prev = cap.read()
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    acc = np.zeros((h, w), np.float32)
    first = np.full((h, w), -1, np.int32)
    idx = 2
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        cg = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        M = estimate_global_motion(prev_gray, cg)
        ref = cv2.warpAffine(prev_gray, M, (w, h)) if M is not None else prev_gray
        diff = cv2.absdiff(ref, cg).astype(np.float32)
        np.maximum(acc, diff, out=acc)
        if idx >= START:
            newly = (diff > NOISE_THRESHOLD) & (first < 0)
            first[newly] = idx
        prev_gray = cg
        idx += 1
    cap.release()
    return acc, first


def load_shots_px():
    hp = CACHE / "homografia.npz"
    if not hp.exists():
        return None
    H = np.load(hp)["H"]
    XY = [(float(r[1]), float(r[2])) for r in
          list(csv.reader(open(ROOT / "debug" / "Secuencia (2).csv")))[1:]
          if len(r) >= 5 and r[4].strip().replace('.', '').isdigit()]
    P = np.array(XY)
    p = np.column_stack([P, np.ones(len(P))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acc, first = one_pass(CLIP)
    valid = first >= 0
    h, w = acc.shape

    # --- normalizaciones ---
    m = np.percentile(acc[acc > 0], 99.7)
    inten = np.clip(acc / (m + 1e-6), 0, 1)                 # 0..1
    lo, hi = first[valid].min(), first[valid].max()
    tnorm = np.zeros((h, w), np.float32)
    tnorm[valid] = (first[valid] - lo) / (hi - lo + 1e-6)   # 0..1 (0=temprano)

    # 1) intensidad (gris)
    gray = (inten ** 0.85 * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "1_intensidad.png"), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # 2) temporal (color, primera llegada)
    tmp = np.zeros((h, w), np.uint8)
    tmp[valid] = (tnorm[valid] * 255).astype(np.uint8)
    temporal = cv2.applyColorMap(tmp, cv2.COLORMAP_TURBO)
    temporal[~valid] = 0
    cv2.imwrite(str(OUT / "2_temporal.png"), temporal)

    # 3) COMBINADA: color = cuando (H), brillo = intensidad (V)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = (120 * (1 - tnorm)).astype(np.uint8)      # azul=temprano rojo=tardio
    hsv[..., 1] = 255
    hsv[..., 2] = (inten ** 0.7 * 255).astype(np.uint8)
    hsv[~valid] = 0
    comb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(OUT / "3_combinada.png"), comb)

    # 4) combinada + malla de tiros (rojo)
    shots = load_shots_px()
    comb_m = comb.copy()
    if shots is not None:
        for x, y in shots:
            cv2.circle(comb_m, (int(x), int(y)), 8, (0, 0, 255), -1)
    cv2.imwrite(str(OUT / "4_combinada_malla.png"), comb_m)

    # 5) pintada (referencia)
    if PAINT.exists():
        shutil.copy(str(PAINT), str(OUT / "5_pintada.png"))

    print(f"[ok] intensidad+temporal en 1 pasada. frames {lo}..{hi}. -> {OUT}")


if __name__ == "__main__":
    main()
