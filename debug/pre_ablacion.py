"""
ABLACION — ¿los pasos 1-4 aportan algo, o basta aplicar la linealidad sobre la
mascara de intensidad que YA teniamos?

Pregunta del usuario (2026-07-27). Se responde midiendo, no teorizando.

Se comparan dos acumulados que difieren SOLO en dos factores:

  A) "ANTIGUO"  como mask_combined.py: cada diff se calcula en el marco de SU
     propio frame (marco MOVIL, encadenado implicito) y SIN normalizar la
     iluminacion. Acumula desde el frame 2.
  B) "NUEVO"    como pre_pasada.py: todos los frames registrados a un marco
     FIJO (el del fondo) y CON normalizacion de iluminacion. Desde START.

Ambos usan el mismo estimador de movimiento, asi que el experimento aisla justo
esas dos variables. Luego se aplica el MISMO realce de linealidad a los dos.

    uv run python debug/pre_ablacion.py
"""
import time

import cv2
import numpy as np

from pre_common import (CLIP, START, CACHE, D_LINEAL, ensure_dirs,
                        estimate_motion, warp_to, to8, side_by_side, grid, save)
import pre_pasada
import pre_linealidad
import pre_render

CACHE_FILE = CACHE / "acc_antiguo.npz"


def acumulado_antiguo(clip=CLIP):
    """Replica el metodo previo: marco movil, sin normalizar iluminacion."""
    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, prev = cap.read()
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                                 (3, 3), 0).astype(np.float32)
    acc = np.zeros((h, w), np.float32)
    idx, t0 = 2, time.time()
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        cg = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY),
                              (3, 3), 0).astype(np.float32)
        M = estimate_motion(prev_gray, cg)          # prev -> curr (marco movil)
        ref = warp_to(prev_gray, M, (h, w)) if M is not None else prev_gray
        np.maximum(acc, np.abs(ref - cg), out=acc)
        prev_gray = cg
        idx += 1
        if idx % 100 == 0:
            print(f"  {idx} ({time.time() - t0:.0f}s)")
    cap.release()
    np.savez_compressed(CACHE_FILE, acc=acc)
    return acc


def main():
    ensure_dirs()
    if CACHE_FILE.exists():
        print("Usando cache acc_antiguo.npz")
        acc_old = np.load(CACHE_FILE)["acc"]
    else:
        print("Recomputando acumulado con el metodo ANTIGUO...")
        acc_old = acumulado_antiguo()

    acc_new = pre_pasada.load()["acc_ff"]

    print("\nAplicando el MISMO realce de linealidad a ambos...")
    lin_old, _, _ = pre_linealidad.lineness(acc_old, 51, 16, mode="disco")
    lin_new, _, _ = pre_linealidad.lineness(acc_new, 51, 16, mode="disco")

    box = pre_render.auto_crop(acc_new, 24)
    cut = lambda im: pre_render.cut(im, box)          # noqa: E731

    save(D_LINEAL / "ablacion_acumulados.png", side_by_side([
        (cut(to8(acc_old, hi_pct=99.5)), "acumulado ANTIGUO (marco movil, sin normalizar)"),
        (cut(to8(acc_new, hi_pct=99.5)), "acumulado NUEVO (marco fijo + normalizado)"),
    ]))
    save(D_LINEAL / "ablacion_linealidad.png", grid([
        (cut(to8(acc_old, hi_pct=99.5)), "1. intensidad ANTIGUA"),
        (cut(to8(acc_new, hi_pct=99.5)), "2. intensidad NUEVA"),
        (cut(to8(lin_old, hi_pct=99.8)), "3. linealidad sobre la ANTIGUA"),
        (cut(to8(lin_new, hi_pct=99.8)), "4. linealidad sobre la NUEVA"),
    ], cols=2))

    # --- medida objetiva: cuanto CONTRASTE linea/fondo logra cada una ---
    # Proxy razonable: que tan lejos esta la cola alta respecto de la mediana.
    print("\n  Contraste de la mascara de linealidad (mayor = mejor separacion):")
    for nom, li in (("ANTIGUA", lin_old), ("NUEVA", lin_new)):
        med = float(np.median(li))
        p999 = float(np.percentile(li, 99.9))
        p99 = float(np.percentile(li, 99))
        mad = float(np.median(np.abs(li - med))) + 1e-6
        print(f"    {nom:<8} p99/mediana={p99/max(med,1e-6):6.1f}   "
              f"(p99.9-mediana)/MAD={(p999-med)/mad:7.1f}")

    for nom, ac in (("ANTIGUA", acc_old), ("NUEVA", acc_new)):
        print(f"    intensidad {nom:<8} media={ac.mean():6.2f}  "
              f"p99={np.percentile(ac, 99):6.1f}")


if __name__ == "__main__":
    main()
