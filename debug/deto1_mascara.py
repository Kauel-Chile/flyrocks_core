"""
DETO1, etapa 1 — el binario por frame y sus detecciones.

Replica el sustrato del software del cliente, segun lo que el nos describio:
saca la mascara mirando la DIFERENCIA DE COLOR ENTRE FRAME Y FRAME (igual que
nosotros), pero la pinta BINARIA (blanco y negro) en vez de en escala de grises.

La diferencia no es cosmetica y conviene tenerla clara antes de comparar
resultados:

    nosotros  ->  guardamos el gris. La intensidad de la estela sobrevive y
                  cualquier decision de umbral se puede rehacer despues.
    deto1     ->  decide el umbral UNA vez, por pixel, al principio de todo.
                  Lo que no lo pasa deja de existir: ninguna etapa posterior
                  lo puede recuperar, por muchas horas que corra.

Eso explica por que ellos PINTABAN encima las trayectorias que no salieron: no
es un capricho de presentacion, es la reparacion manual del recorte del umbral.

Lo que este script produce (etapa cara, una sola lectura del video):

    detecciones.npz     por frame, las componentes conexas del binario
                        (frame, x, y, area, largo, ancho, angulo)
    binario_acum.png    su mascara blanco y negro: el OR de todos los frames
    gris_acum.png       la nuestra, el maximo del z-score, para comparar
    perdidas.png        lo que el gris ve y el binario NO: el costo del umbral

La segunda etapa (deto1_flujo.py) toma detecciones.npz y arma las trayectorias
con asociacion global.

    uv run python debug/deto1_mascara.py [--lo 4] [--hi 8]

Se apoya en el aparato del preprocesamiento que ya estaba hecho: el modelo de
fondo del pre-roll, el registro contra el fondo (sin deriva) y la normalizacion
de iluminacion que sobrevive al flash de la detonacion. Sin eso, el diff entre
frames de un video de dron es casi todo movimiento de camara.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pre_common import (CACHE, SIGMA_FLOOR, START, estimate_motion, warp_to,
                        match_illum, save, to8)
import pre_fondo

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "debug" / "casos" / "3160-789" / "clip.mp4"
OUT = ROOT / "debug" / "out" / "8_deto1"
DETECCIONES = OUT / "detecciones.npz"

# Umbrales en SIGMAS del ruido del diff, no en niveles de gris: el ruido del
# H.264 no es uniforme por el cuadro y un umbral fijo castiga las zonas lisas.
# 4/8 sale del contact sheet de pre_umbrales.py, donde la histeresis 4->8 era la
# que conservaba estelas tenues sin derramarse en la zona del blast.
LO_DEF = 4.0
HI_DEF = 8.0

# Una componente de menos de 6 px es ruido de compresion: a 4K, una roca deja
# una estela de decenas de pixeles por frame.
AREA_MIN = 6
# Tope de area por componente. Por encima de esto no es una roca sino el frente
# de polvo; se conserva su nucleo fuerte (ver hysteresis_guarded) pero no se
# emite como deteccion, porque su centroide no significa nada.
AREA_MAX = 4000


def hysteresis_guarded(img, lo, hi, max_area=AREA_MAX):
    """Doble umbral con salvaguarda contra derrame. Igual que pre_umbrales.

    La histeresis pura se derrama: en la zona del blast el umbral bajo conecta
    todo en un unico blob. Aca, la componente con semilla se acepta entera solo
    si es chica; si es un derrame, se conserva unicamente su nucleo fuerte.
    """
    weak = (img > lo).astype(np.uint8)
    strong = img > hi
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    if n <= 1:
        return np.zeros(img.shape, bool)
    has_seed = np.zeros(n, bool)
    has_seed[np.unique(lab[strong])] = True
    has_seed[0] = False
    areas = stats[:, cv2.CC_STAT_AREA]
    out = (has_seed & (areas <= max_area))[lab]
    out |= (has_seed & (areas > max_area))[lab] & strong
    return out


def detecciones_de(mask, frame_idx):
    """Componentes conexas del binario -> una fila por deteccion.

    Se guarda la forma (largo, ancho, angulo) ademas del centroide porque la
    asociacion global la va a necesitar: una estela es un segmento orientado, y
    su direccion dice hacia donde sigue la roca mejor que dos centroides
    consecutivos.

    OJO con la implementacion: la version obvia es `np.nonzero(lab == i)` por
    componente, y a 4K eso recorre los 8,3 millones de pixeles del cuadro UNA
    VEZ POR COMPONENTE. Con miles de componentes por frame el script se cuelga
    sin dar señales (medido: no termino un solo frame en 15 minutos). Aca cada
    componente se mide dentro de su propio bounding box, que es lo que ya
    devuelve connectedComponentsWithStats: el costo total pasa a ser del orden
    de los pixeles encendidos.
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    filas = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < AREA_MIN or area > AREA_MAX:
            continue
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        sub = (lab[y0:y0 + h, x0:x0 + w] == i).astype(np.uint8)
        m = cv2.moments(sub, binaryImage=True)
        if m["m00"] <= 0:
            continue
        # Momentos centrales normalizados: la elipse equivalente de la mancha.
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]
        comun = np.sqrt(max(4.0 * mu11 * mu11 + (mu20 - mu02) ** 2, 0.0))
        l1 = (mu20 + mu02 + comun) / 2.0
        l2 = (mu20 + mu02 - comun) / 2.0
        largo = 4.0 * np.sqrt(max(l1, 0.0))
        ancho = 4.0 * np.sqrt(max(l2, 0.0))
        ang = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
        filas.append((frame_idx, cent[i][0], cent[i][1], area,
                      float(largo), float(ancho), float(ang)))
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--clip", default=str(CLIP))
    ap.add_argument("--lo", type=float, default=LO_DEF)
    ap.add_argument("--hi", type=float, default=HI_DEF)
    ap.add_argument("--start", type=int, default=START,
                    help="frame de la primera detonacion; antes es pre-roll")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hasta", type=int, default=0,
                    help="cortar en este frame (0 = todo). Para probar rapido.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if DETECCIONES.exists() and not args.force:
        z = np.load(DETECCIONES)
        print(f"ya existe {DETECCIONES.name}: {len(z['det'])} detecciones "
              f"(--force para rehacer)")
        return

    fondo = pre_fondo.load()
    bg_mean = fondo["bg_mean"].astype(np.float32)
    diff_sigma = np.maximum(fondo["diff_sigma"], SIGMA_FLOOR).astype(np.float32)
    ref_med, ref_mad = float(fondo["ref_med"]), float(fondo["ref_mad"])
    bg_u8 = np.clip(bg_mean, 0, 255).astype(np.uint8)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"no se pudo abrir {args.clip}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shape = (h, w)
    ones = np.ones(shape, np.float32)

    acum_bin = np.zeros(shape, bool)         # la mascara de ellos
    acum_gris = np.zeros(shape, np.float32)  # la nuestra (max del z-score)
    filas = []
    prev_al = None
    M_prev = None
    n_fallback = 0
    idx = 0
    t0 = time.time()

    print(f"DETO1 etapa 1: {nfr} frames {w}x{h}, histeresis {args.lo}->{args.hi} sigmas", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             (3, 3), 0).astype(np.float32)

        M = estimate_motion(g, bg_u8)
        if M is None:
            M = M_prev                 # el humo tapo demasiado: reusa el anterior
            n_fallback += 1
        else:
            M_prev = M
        if M is None:
            prev_al = None
            idx += 1
            continue

        g_al = warp_to(g, M, shape)
        valid = warp_to(ones, M, shape) > 0.99
        g_al = match_illum(g_al, ref_med, ref_mad, mask=valid)
        g_al[~valid] = bg_mean[~valid]

        if prev_al is not None and idx >= args.start:
            z_ff = np.abs(g_al - prev_al) / diff_sigma
            mask = hysteresis_guarded(z_ff, args.lo, args.hi)
            acum_bin |= mask
            np.maximum(acum_gris, z_ff, out=acum_gris)
            filas.extend(detecciones_de(mask, idx))

        prev_al = g_al
        idx += 1
        if args.hasta and idx >= args.hasta:
            print(f"  corte en el frame {idx} (--hasta)")
            break
        if idx % 25 == 0:
            print(f"  {idx}/{nfr}  {len(filas)} detecciones  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    cap.release()
    det = np.array(filas, np.float32) if filas else np.zeros((0, 7), np.float32)
    np.savez_compressed(DETECCIONES, det=det, shape=np.array(shape),
                        umbrales=np.array([args.lo, args.hi]),
                        start=args.start, nframes=idx)

    save(OUT / "binario_acum.png", (acum_bin * 255).astype(np.uint8))
    save(OUT / "gris_acum.png", to8(acum_gris, hi_pct=99.5))

    # Lo que el gris ve y el binario no. Es la medida directa de lo que cuesta
    # decidir el umbral al principio: cada pixel rojo es estela que deto1 tiene
    # que reponer pintando a mano.
    gris8 = to8(acum_gris, hi_pct=99.5)
    perdido = (gris8 > 40) & (~acum_bin)
    vis = cv2.cvtColor(gris8, cv2.COLOR_GRAY2BGR)
    vis[perdido] = (0, 0, 255)
    vis[acum_bin] = (0, 255, 0)
    save(OUT / "perdidas.png", vis)

    print(f"\n  {len(det)} detecciones en {idx} frames "
          f"({time.time() - t0:.0f}s, {n_fallback} sin registro propio)")
    print(f"  binario cubre {acum_bin.mean() * 100:.3f}% del cuadro")
    print(f"  el gris ve {perdido.sum() / max(acum_bin.sum(), 1) * 100:.1f}% mas "
          f"superficie de estela que el binario")
    print(f"  -> {DETECCIONES}")


if __name__ == "__main__":
    main()
