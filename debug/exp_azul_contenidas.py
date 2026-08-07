"""
Ejercicio: por cada trazo AZUL, cuales trayectorias de 1_dedup van
MAYORITARIAMENTE dentro (siguen el trazo), vs las que solo lo ROZAN.

El discriminador es la FRACCION del largo de la traza que cae dentro del azul.
Asi deberiamos ver, en el caso de la parabola, los ~2 pedazos (va + vuelve).
"""
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from clean_and_stitch import load_tracks, dedup                      # noqa: E402
from paint_filter import PaintMask                                   # noqa: E402

OUT = ROOT / "debug" / "out" / "6_mascaras" / "trayectorias"
GRAY = ROOT / "debug" / "out" / "6_mascaras" / "1_intensidad.png"
PAINT = ROOT / "debug" / "out" / "4_fase0_referencias" / "lienzo_para_pintar_gris - Copy.png"
FRAC = 0.5                       # fraccion minima del largo dentro del azul

COLORS = [(0, 255, 255), (0, 128, 255), (0, 255, 0), (255, 0, 255),
          (255, 255, 0), (0, 0, 255)]

tracks, origin = load_tracks()
base = dedup(tracks, origin, 0.03)
pm = PaintMask(PAINT)
nlab, blue_lab = cv2.connectedComponents(pm.blue.astype(np.uint8))
h, w = blue_lab.shape

img = cv2.imread(str(GRAY))
ov = img.copy(); ov[pm.blue] = (110, 60, 0)
img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)

asignadas = defaultdict(list)
for a in base:
    px = np.clip(a[:, 0].astype(int), 0, w - 1)
    py = np.clip(a[:, 1].astype(int), 0, h - 1)
    labs = blue_lab[py, px]
    # fraccion dentro de cada trazo
    best_k, best_f = 0, 0.0
    for k in range(1, nlab):
        f = float((labs == k).mean())
        if f > best_f:
            best_f, best_k = f, k
    if best_f >= FRAC:
        asignadas[best_k].append((a, best_f))
    else:
        cv2.polylines(img, [a[:, :2].astype(np.int32).reshape(-1, 1, 2)],
                      False, (70, 70, 70), 1, cv2.LINE_AA)   # solo roza -> tenue

print(f"1_dedup: {len(base)} trayectorias")
for k in range(1, nlab):
    lst = asignadas.get(k, [])
    col = COLORS[(k - 1) % len(COLORS)]
    for a, f in lst:
        cv2.polylines(img, [a[:, :2].astype(np.int32).reshape(-1, 1, 2)],
                      False, col, 4, cv2.LINE_AA)
    fr = ", ".join(f"{f:.0%}" for _, f in lst)
    print(f"   azul#{k} ({['cyan','azul','verde','magenta','amarillo','rojo'][(k-1)%6]}): "
          f"{len(lst)} trayectorias MAYORITARIAMENTE dentro  [fracciones: {fr}]")

cv2.putText(img, f"trayectorias de 1_dedup con >={FRAC:.0%} de su largo dentro del azul",
            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
cv2.imwrite(str(OUT / "exp_azul_contenidas.png"), img)
print(f"-> {OUT / 'exp_azul_contenidas.png'}")
