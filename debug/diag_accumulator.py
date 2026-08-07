"""
Diagnostico: reproduce el acumulador de intensidades (estilo mascara del cliente)
y compara TODO el movimiento vs. lo que el filtro de energia (percentil 96)
conserva. Si las estelas de roca desaparecen al aplicar el percentil, confirma
que el filtro de energia esta botando las rocas y quedandose con el humo.

    uv run python debug/diag_accumulator.py
"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from harness import estimate_global_motion, CLIP_PATH, NOISE_THRESHOLD, OUT_DIR

DIAG_DIR = OUT_DIR / "0_diagnostico"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
ENERGY_PCT = 96.0


def build_accumulators(clip_path):
    cap = cv2.VideoCapture(str(clip_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, prev = cap.read()
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    acc_all = np.zeros((h, w), dtype=np.float32)   # max de todo el movimiento
    intens = []                                     # intensidades de eventos
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        M = estimate_global_motion(prev_gray, curr_gray)
        ref = cv2.warpAffine(prev_gray, M, (w, h)) if M is not None else prev_gray
        diff = cv2.absdiff(ref, curr_gray).astype(np.float32)
        np.maximum(acc_all, diff, out=acc_all)
        v = diff[diff > NOISE_THRESHOLD]
        if v.size:
            intens.append(v)
        prev_gray = curr_gray
    cap.release()

    thr96 = float(np.percentile(np.concatenate(intens), ENERGY_PCT))
    acc_kept = np.where(acc_all >= thr96, acc_all, 0).astype(np.float32)
    print(f"  Umbral percentil {ENERGY_PCT}: intensidad >= {thr96:.1f}")
    frac = (acc_all >= thr96).sum() / (acc_all > NOISE_THRESHOLD).sum()
    print(f"  Pixeles con movimiento que SOBREVIVEN al percentil: {frac*100:.1f}%")
    return acc_all, acc_kept, thr96


def to_png(acc, path, gamma=0.6, clip_pct=99.7, colormap="gray"):
    # normaliza por percentil alto (no por el max) para no aplastar las estelas
    pos = acc[acc > 0]
    m = np.percentile(pos, clip_pct) if pos.size else (acc.max() + 1e-6)
    norm = (np.clip(acc / (m + 1e-6), 0, 1) ** gamma * 255).astype(np.uint8)
    if colormap == "gray":
        img = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.applyColorMap(norm, cv2.COLORMAP_BONE)
    cv2.imwrite(str(path), img)
    print(f"  -> {path}")


ACC_CACHE = OUT_DIR / "_cache"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=str, default=str(CLIP_PATH))
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--clip-pct", type=float, default=99.7)
    ap.add_argument("--colormap", choices=["gray", "bone"], default="gray")
    ap.add_argument("--render-only", action="store_true",
                    help="re-renderiza desde el acumulador cacheado (instantaneo)")
    cfg = ap.parse_args()
    ACC_CACHE.mkdir(parents=True, exist_ok=True)
    npy = ACC_CACHE / "acc_all.npy"

    if cfg.render_only and npy.exists():
        acc_all = np.load(npy)
    else:
        acc_all, acc_kept, thr = build_accumulators(Path(cfg.clip))
        np.save(npy, acc_all)
    to_png(acc_all, DIAG_DIR / "movimiento_todo.png",
           gamma=cfg.gamma, clip_pct=cfg.clip_pct, colormap=cfg.colormap)


if __name__ == "__main__":
    main()
