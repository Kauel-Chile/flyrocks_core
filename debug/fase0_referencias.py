"""
Fase 0 — genera las imagenes de referencia para que el usuario entregue inputs.

Salidas en out/4_fase0_referencias/:
  1. contact_sheet_inicio.png  -> mosaico de frames 13.0-16.0s etiquetados con el
     tiempo. El usuario indica el tiempo/frame EXACTO donde parte la detonacion.
  2. frame_full_res_13s.png    -> un frame a resolucion completa (4K) para buscar
     coordenadas de tiros en Paint.
  3. esquema_malla_tiros.png   -> el patron de tiros del CSV con sus numeros,
     coloreado por tiempo de detonacion, para identificar 'tiro XXX'.

    uv run python debug/fase0_referencias.py
"""
import sys
import csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "debug" / "Video de cliente 3160-789.mp4"
CSV = ROOT / "debug" / "Secuencia (2).csv"
OUT = ROOT / "debug" / "out" / "4_fase0_referencias"
OUT.mkdir(parents=True, exist_ok=True)


def label(img, text):
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255), 2)


def contact_sheet(t0=13.0, t1=16.0, step_s=0.1, cols=6, tw=520, th=293):
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    times = np.arange(t0, t1 + 1e-6, step_s)
    thumbs = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ret, fr = cap.read()
        if not ret:
            continue
        th_img = cv2.resize(fr, (tw, th))
        label(th_img, f"t={t:.2f}s  (f{int(round(t*fps))})")
        thumbs.append(th_img)
    cap.release()
    rows = int(np.ceil(len(thumbs) / cols))
    grid = np.zeros((rows * th + (rows + 1) * 6, cols * tw + (cols + 1) * 6, 3),
                    np.uint8)
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        y = 6 + r * (th + 6)
        x = 6 + c * (tw + 6)
        grid[y:y + th, x:x + tw] = im
    cv2.imwrite(str(OUT / "contact_sheet_inicio.png"), grid)
    print(f"  contact sheet: {len(thumbs)} frames ({t0}-{t1}s)")


def full_res_frame(t=13.0):
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ret, fr = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(str(OUT / "frame_full_res_13s.png"), fr)
        print(f"  frame full-res {fr.shape[1]}x{fr.shape[0]} @ {t}s")


def mesh_schematic():
    X, Y, T, L = [], [], [], []
    for r in list(csv.reader(open(CSV)))[1:]:
        if len(r) >= 5 and r[4].strip().replace('.', '').isdigit():
            X.append(float(r[1])); Y.append(float(r[2]))
            T.append(float(r[4])); L.append(r[0].strip())
    X, Y, T = np.array(X), np.array(Y), np.array(T)

    fig, ax = plt.subplots(figsize=(20, 14))
    sc = ax.scatter(X, Y, c=T, cmap="jet", s=120, edgecolors="k", linewidths=0.4)
    for x, y, lab in zip(X, Y, L):
        ax.annotate(lab, (x, y), fontsize=6, ha="center", va="center",
                    color="white")
    ie, il = int(np.argmin(T)), int(np.argmax(T))
    for i, tag in [(ie, "PRIMERO"), (il, "ULTIMO")]:
        ax.annotate(f"{L[i]}\n({tag})", (X[i], Y[i]), fontsize=10, weight="bold",
                    color="black", xytext=(15, 15), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->"))
    plt.colorbar(sc, label="tiempo de detonacion (ms)  [azul=primero, rojo=ultimo]")
    ax.set_title("Malla de tiros (CSV) — orientacion aprox. En el video: primero "
                 "abajo-derecha, propaga izquierda/arriba", fontsize=13)
    ax.set_xlabel("X (mina)"); ax.set_ylabel("Y (mina)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "esquema_malla_tiros.png", dpi=110)
    print(f"  esquema malla: {len(X)} tiros")


if __name__ == "__main__":
    print("[fase0] generando referencias...")
    contact_sheet()
    full_res_frame()
    mesh_schematic()
    print(f"[fase0] listo -> {OUT}")
