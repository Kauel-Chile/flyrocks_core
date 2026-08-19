"""
DETO1, barrido — que combinacion de parametros produce ROCAS y no culebras.

La primera corrida completa dejo un ovillo dentro de la nube de polvo en vez del
abanico radial de estelas que se ve en la mascara del cliente. Antes de seguir
tocando el algoritmo hay que saber si el problema es el punto de operacion o el
planteamiento, y eso se contesta barriendo, no adivinando de a un parametro.

La metrica que manda es la RECTITUD (neto/recorrido). Un vuelo balistico da
~0.9; un camino que serpentea entre manchas de polvo da ~0.2. El numero de
trayectorias por si solo no dice nada: 600 culebras se ven igual de mal que 16.

    uv run python debug/deto1_barrido.py

Deja debug/out/8_deto1/barrido.csv con una fila por combinacion, ordenable, y
un PNG por cada una de las mejores para mirarlas de una sola pasada.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pre_common import save
import deto1_flujo as F

OUT = F.OUT
CSV = OUT / "barrido.csv"
GALERIA = OUT / "galeria"

# El espacio a explorar. Son ~54 combinaciones y cada una tarda entre 20 s y un
# minuto, asi que el barrido entero cabe holgado en una noche.
REJILLA = dict(
    elong=[3.0, 4.0, 6.0],          # cuan alargada tiene que ser una estela
    dmin=[15.0, 30.0, 50.0],        # velocidad minima: el polvo es lento
    entrada=[1.0, 1.5, 2.5],        # cuanto cuesta abrir una trayectoria
    tol_sector=[1, 2],              # cuanto puede doblar entre enlaces
)


# Tope de arcos por corrida. Esta maquina trabaja con ~1 GB de RAM libre y cada
# arco ocupa unos 32 bytes en los arrays del DP, que ademas se duplican durante
# el ordenamiento. El primer intento de barrido arranco por la combinacion mas
# densa (elong=3 deja 151.217 detecciones, ~37 millones de arcos) y el proceso
# murio sin completar UNA sola combinacion ni alcanzar a escribir el CSV.
MAX_ARCOS = 12_000_000


def estimar_arcos(det, gap, vmax, muestra=300):
    """Cuantos arcos tendria el grafo, sin construirlo.

    Se muestrean detecciones y se cuentan sus vecinos reales con el KD-tree: la
    formula por densidad subestima mucho, porque las detecciones estan apretadas
    en la zona del blast y vacias en el resto del cuadro.
    """
    from scipy.spatial import cKDTree
    frames = det[:, 0].astype(np.int64)
    xy = det[:, 1:3]
    por_frame = {}
    for f in np.unique(frames):
        por_frame[int(f)] = np.nonzero(frames == f)[0]
    arboles = {f: cKDTree(xy[i]) for f, i in por_frame.items()}

    rng = np.random.default_rng(0)
    idx = rng.choice(len(det), size=min(muestra, len(det)), replace=False)
    total = 0
    for i in idx:
        f = int(frames[i])
        for d in range(1, gap + 1):
            t = arboles.get(f + d)
            if t is not None:
                total += len(t.query_ball_point(xy[i], r=vmax * d))
    return int(total / len(idx) * len(det))


def una_corrida(det_todas, shape, elong, dmin, entrada, tol_sector, gap=4):
    largo, ancho = det_todas[:, 4], np.maximum(det_todas[:, 5], 0.5)
    det = det_todas[(largo / ancho >= elong) & (det_todas[:, 3] >= 25.0)]
    if len(det) < 100:
        return None, None

    # Se baja el gap hasta que el grafo quepa, en vez de saltar la combinacion:
    # una corrida con gap 2 dice algo, y no correrla no dice nada.
    while gap >= 1 and estimar_arcos(det, gap, F.VMAX_PX) > MAX_ARCOS:
        gap -= 1
    if gap < 1:
        return "pesada", None

    F.DMIN_PX = dmin        # el filtro duro vive en el modulo; se pisa por corrida
    arcos = F.construir_arcos(det, gap, F.VMAX_PX)
    if len(arcos[0]) == 0:
        return None, None

    usada = np.zeros(len(det), bool)
    trayectorias = []
    while len(trayectorias) < 20000:
        caminos = F.dp_mejores(det, arcos, usada, 400, tol_sector, entrada)
        if not caminos:
            break
        nuevas = 0
        for _, camino in caminos:
            usada[camino] = True
            if len(camino) >= 4:
                trayectorias.append(camino)
                nuevas += 1
        if nuevas == 0:
            break
    res = F.resumen(det, trayectorias, usada)
    res["gap"] = gap
    res["detecciones"] = len(det)
    return res, (det, trayectorias)


def main():
    z = np.load(F.DETECCIONES)
    det_todas, shape = z["det"], tuple(z["shape"])
    GALERIA.mkdir(parents=True, exist_ok=True)

    # De liviano a pesado: si el barrido se corta, lo que alcanzo a correr son
    # las combinaciones baratas, que ya dicen bastante.
    combos = sorted(itertools.product(*REJILLA.values()), key=lambda c: -c[0])
    print(f"{len(combos)} combinaciones sobre {len(det_todas)} detecciones crudas",
          flush=True)

    filas = []
    t0 = time.time()
    for k, (elong, dmin, entrada, tol) in enumerate(combos, 1):
        t1 = time.time()
        try:
            res, datos = una_corrida(det_todas, shape, elong, dmin, entrada, tol)
        except MemoryError:
            print(f"  [{k}/{len(combos)}] elong={elong} dmin={dmin} "
                  f"entrada={entrada} tol={tol}: SIN MEMORIA, se salta",
                  flush=True)
            continue
        if res == "pesada":
            print(f"  [{k}/{len(combos)}] elong={elong} dmin={dmin} "
                  f"entrada={entrada} tol={tol}: grafo sobre el tope, se salta",
                  flush=True)
            continue
        if res is None:
            print(f"  [{k}/{len(combos)}] elong={elong} dmin={dmin} "
                  f"entrada={entrada} tol={tol}: sin material", flush=True)
            continue
        res.update(elong=elong, dmin=dmin, entrada=entrada, tol_sector=tol,
                   segundos=round(time.time() - t1, 1))
        filas.append(res)
        print(f"  [{k}/{len(combos)}] elong={elong} dmin={dmin} entrada={entrada} "
              f"tol={tol} -> {res['n']} tray ({res['largas']} largas), "
              f"rect {res['rectitud']:.2f} / largas {res['rect_largas']:.2f}, "
              f"SCORE {res['score']:.0f} ({res['segundos']:.0f}s)", flush=True)

        # Solo se dibujan las prometedoras: rectas y con material suficiente.
        if res["score"] >= 10:
            d, trs = datos
            nombre = (f"s{res['score']:.0f}_r{res['rect_largas']:.2f}_elong{elong}_"
                      f"dmin{dmin:.0f}_ent{entrada}_tol{tol}.png")
            save(GALERIA / nombre, F.pintar(d, trs, shape))

        # El CSV se reescribe en cada vuelta: si esto se corta, lo corrido sirve.
        if filas:
            cols = ["score", "rect_largas", "largas", "rectitud", "n",
                    "mediana", "cobertura", "elong", "dmin", "entrada",
                    "tol_sector", "gap", "detecciones", "segundos"]
            orden = sorted(filas, key=lambda r: -r["score"])
            with open(CSV, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                for r in orden:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")

    print(f"\n  {len(filas)} corridas en {(time.time() - t0) / 60:.0f} min")
    if filas:
        mejor = max(filas, key=lambda r: r["score"])
        print(f"  mejor score {mejor['score']:.0f} "
              f"({mejor['largas']} largas a rectitud {mejor['rect_largas']:.2f}) con "
              f"elong={mejor['elong']} dmin={mejor['dmin']} "
              f"entrada={mejor['entrada']} tol={mejor['tol_sector']} "
              f"({mejor['n']} trayectorias)")
        print(f"  -> {CSV}")
        print(f"  -> {GALERIA} (solo las prometedoras)")


if __name__ == "__main__":
    main()
