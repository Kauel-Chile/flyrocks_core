"""
Diagnostico simple: las trayectorias de 1_dedup (174) con la clase AZUL
superpuesta SEMI-TRANSPARENTE, para ver si las trayectorias pasan o no por
dentro de los trazos azules marcados.
"""
import sys
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

tracks, origin = load_tracks()
base = dedup(tracks, origin, 0.03)              # las mismas 174 de 1_dedup
print(f"trayectorias 1_dedup: {len(base)}")

img = cv2.imread(str(GRAY))
if img is None:
    img = np.zeros((2160, 3840, 3), np.uint8)

pm = PaintMask(PAINT)
nlab, blue_lab = cv2.connectedComponents(pm.blue.astype(np.uint8))
h, w = pm.blue.shape

# 1) azul semi-transparente DEBAJO (para no tapar las trazas)
ov = img.copy()
ov[pm.blue] = (255, 120, 0)
img = cv2.addWeighted(ov, 0.40, img, 0.60, 0)

# 2) clasificar: que trayectorias pasan por dentro de algun azul
por_trazo = {k: 0 for k in range(1, nlab)}
dentro, fuera = [], []
for a in base:
    px = np.clip(a[:, 0].astype(int), 0, w - 1)
    py = np.clip(a[:, 1].astype(int), 0, h - 1)
    labs = blue_lab[py, px]
    hit = labs[labs > 0]
    if hit.size:
        dentro.append(a)
        for k in np.unique(hit):
            por_trazo[int(k)] += 1
    else:
        fuera.append(a)

# 3) dibujar: las que NO tocan azul, tenues; las que SI, amarillo grueso
for a in fuera:
    cv2.polylines(img, [a[:, :2].astype(np.int32).reshape(-1, 1, 2)],
                  False, (90, 90, 90), 2, cv2.LINE_AA)
for a in dentro:
    cv2.polylines(img, [a[:, :2].astype(np.int32).reshape(-1, 1, 2)],
                  False, (0, 255, 255), 4, cv2.LINE_AA)

print(f"de las {len(base)}: {len(dentro)} PASAN por algun azul, {len(fuera)} no")
for k, n in por_trazo.items():
    print(f"   azul#{k}: {n} trayectorias de 1_dedup pasan por dentro")

cv2.putText(img, "amarillo = trayectorias de 1_dedup que SI pasan por el azul",
            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
cv2.imwrite(str(OUT / "exp_1dedup_con_azul.png"), img)
print(f"-> {OUT / 'exp_1dedup_con_azul.png'}")
