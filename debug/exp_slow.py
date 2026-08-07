"""Experimento: todas - rapidas = LENTAS. Para ver que queda (parabolas lentas
+ polvo). Umbral de rapida = 25 px/frame (radial)."""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug"))
from clean_and_stitch import load_tracks, dedup, render

OUT = ROOT / "debug" / "out" / "6_mascaras" / "trayectorias"
GRAY = ROOT / "debug" / "out" / "6_mascaras" / "1_intensidad.png"
THRESH = 25.0

tracks, origin = load_tracks()
bg = cv2.imread(str(GRAY))


def rspeed(a):
    r = np.hypot(a[:, 0] - origin[0], a[:, 1] - origin[1])
    dt = a[:, 2].max() - a[:, 2].min()
    return (r.max() - r.min()) / (dt + 1)


speeds = np.array([rspeed(a) for a in tracks])
slow = [a for a, s in zip(tracks, speeds) if s < THRESH]
render(slow, origin, OUT / "exp_lentas_todas.png", bg)
d = dedup(slow, origin, 0.03)
render(d, origin, OUT / "exp_lentas_dedup.png", bg)
print(f"todas {len(tracks)} - rapidas = lentas {len(slow)} | dedup {len(d)}")
