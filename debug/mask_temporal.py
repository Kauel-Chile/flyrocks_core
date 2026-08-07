"""
Mascara TEMPORAL: cada pixel se colorea por el PRIMER frame en que le llega
movimiento (primera llegada). azul=temprano -> rojo=tardio (colormap TURBO).

Idea: la roca (rapida) llega primero a un pixel; el polvo (lento) despues. Asi
el color separa el borde de ataque de la roca del polvo que la sigue, muestra la
secuencia de la tronadura, y da colores distintos a estelas que se cruzan en
tiempos distintos (ayuda a separar cruces).

Se genera sobre la misma ventana que la mascara gris (clip 26s) para comparar
lado a lado.

    uv run python debug/mask_temporal.py
"""
import sys
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
CLIP = SCRATCH / "clip_blast_4k_long.mp4"      # misma ventana que la gris (26s)
START = 48                                      # frame de detonacion (ignora deriva previa)
OUTDIR = ROOT / "debug" / "out" / "0_diagnostico" / "comparacion_4k"


def build_first_arrival(clip):
    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, prev = cap.read()
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    first = np.full((h, w), -1, dtype=np.int32)
    idx = 2
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        M = estimate_global_motion(prev_gray, curr_gray)
        ref = cv2.warpAffine(prev_gray, M, (w, h)) if M is not None else prev_gray
        diff = cv2.absdiff(ref, curr_gray)
        if idx >= START:
            newly = (diff > NOISE_THRESHOLD) & (first < 0)
            first[newly] = idx
        prev_gray = curr_gray
        idx += 1
    cap.release()
    return first, idx


def main():
    first, nframes = build_first_arrival(CLIP)
    valid = first >= 0
    print(f"Pixeles con movimiento (desde frame {START}): {int(valid.sum()):,}")

    lo, hi = first[valid].min(), first[valid].max()
    norm = np.zeros(first.shape, np.uint8)
    norm[valid] = ((first[valid] - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0

    # leyenda
    cv2.putText(color, "azul = temprano   rojo = tardio  (primera llegada)",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "6_MI_mascara_temporal.png"
    cv2.imwrite(str(out), color)
    print(f"[viz] -> {out}  (frames {lo}..{hi})")


if __name__ == "__main__":
    main()
