"""
DETO1, etapa 2 — asociacion GLOBAL de las detecciones en trayectorias.

Esta es la parte que hace distinto al software del cliente. Nuestro pipeline
asocia frame a frame (Kalman + hungaro): decide con lo que ve en el momento y
no puede desdecirse, asi que una mala asignacion en el frame 120 se arrastra
hasta el final. De ahi la fragmentacion y los duplicados.

Aca el problema se plantea entero de una vez: TODAS las detecciones de TODOS los
frames en un grafo, y se buscan las trayectorias que minimizan un costo global.
Una asignacion que parece buena localmente pero deja el resto sin explicacion no
se elige. Es lo que el cliente compraba con sus ~6 horas.

COMO, en concreto: DP greedy de caminos mas cortos sucesivos sobre el grafo
temporal (Pirsiavash et al., 2011). El grafo es un DAG —los arcos van siempre
hacia adelante en el tiempo—, asi que el mejor camino se calcula de una pasada
en orden de frame. Se extrae el mejor, se marcan sus detecciones como usadas y
se repite mientras el camino tenga costo negativo, es decir, mientras explique
mas de lo que cuesta abrir una trayectoria nueva.

  costo de un camino = ENTRADA + SALIDA - (premio por deteccion) x n
                       + suma de los costos de enlace

Se prefiere esto al min-cost flow exacto (OR-Tools, ya instalado) por una razon
practica: el greedy da el mismo tipo de resultado, corre en una fraccion del
tiempo y se puede cortar en cualquier momento con lo encontrado hasta ahi. El
exacto queda como paso siguiente si el greedy muestra que vale la pena.

EL HUECO ES LA GRACIA. Los enlaces pueden saltar hasta --gap frames, asi que una
estela que se apaga tres frames y reaparece sigue siendo UNA trayectoria. Eso es
—en automatico— lo que el cliente hacia pintando encima las trayectorias que su
binario habia cortado.

    uv run python debug/deto1_flujo.py

Entra debug/out/8_deto1/detecciones.npz (lo deja deto1_mascara.py) y salen las
trayectorias, un PNG con ellas pintadas y otro contra la mascara del cliente.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pre_common import save

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "debug" / "out" / "8_deto1"
DETECCIONES = OUT / "detecciones.npz"
TRAYECTORIAS = OUT / "trayectorias.npz"

# Velocidad maxima admitida, en pixeles por frame. A 4K las rocas rapidas saltan
# 30-60 px/frame (medido); el tope va holgado porque cortarlo es exactamente el
# error que rompia las trazas en el tracker hungaro (max_dist=30 hardcodeado).
VMAX_PX = 140.0

# Velocidad MINIMA, y es un filtro DURO, no una penalizacion. Una roca en vuelo
# se mueve; el polvo se queda casi donde esta. Con el piso blando (primera
# version) el DP igual encadenaba manchas de humo vecinas a lo largo de todo el
# video: enlaces de 10 px alineados costaban ~0.1 y cada deteccion pagaba 1, asi
# que alargar siempre convenia. Resultado medido: trayectorias de 247-268
# detecciones de mediana sobre un clip de 405 frames — "rocas" volando nueve
# segundos. A 4K una roca salta 30-60 px por frame; 15 deja margen de sobra.
DMIN_PX = 15.0

# Costos. Estan en la misma escala: un enlace perfecto cuesta ~0 y el premio por
# explicar una deteccion es 1. Con ENTRADA=2.5, una trayectoria necesita ~6
# detecciones bien enlazadas para pagar su apertura y su cierre.
#
# La clave es que un enlace MEDIOCRE tiene que costar mas de 1: si no, alargar
# siempre conviene y el resultado es una sola culebra por region.
PREMIO_DET = 1.0
ENTRADA = 2.5
SALIDA = 2.5
# OJO con W_DIST: penalizar la distancia recorrida es penalizar a las rocas
# RAPIDAS, que son precisamente las peligrosas y las que hay que conservar. Con
# W_DIST=2 no salia ninguna trayectoria: un enlace bueno costaba mas que el
# premio por explicar la deteccion. Queda casi en cero a proposito — lo que debe
# costar no es ir lejos, es la INCOHERENCIA (doblar, o no calzar con la estela).
W_DIST = 0.3      # cuanto pesa alejarse
W_GAP = 0.45      # cuanto cuesta cada frame saltado
W_ANG = 1.5       # cuanto cuesta doblar respecto a la direccion de las estelas
W_VEL = 1.5       # cuanto cuesta que el salto no calce con el largo de la estela

# Sectores en que se cuantiza la direccion de un enlace. 16 sectores = 22,5
# grados cada uno: con tolerancia de +-1 sector, una trayectoria puede doblar
# hasta ~45 grados entre enlaces, que es holgado para un vuelo balistico y
# estrecho para impedir el rebote.
N_SECTORES = 16

INF = 1e18


def construir_arcos(det, gap, vmax, w_dist=None, w_ang=None, w_vel=None):
    """Para cada deteccion, sus posibles continuaciones en los frames siguientes.

    Se usa un KD-tree por frame en vez de comparar todo contra todo: con ~150
    detecciones por frame y 400 frames, la version ingenua son 3.500 millones de
    pares y esta son unos pocos millones.
    """
    w_dist = W_DIST if w_dist is None else w_dist
    w_ang = W_ANG if w_ang is None else w_ang
    w_vel = W_VEL if w_vel is None else w_vel
    frames = det[:, 0].astype(np.int64)
    xy = det[:, 1:3]
    ang = det[:, 6]
    largo = det[:, 4]
    orden = np.argsort(frames, kind="stable")
    det_por_frame = {}
    for i in orden:
        det_por_frame.setdefault(int(frames[i]), []).append(i)
    for f in det_por_frame:
        det_por_frame[f] = np.array(det_por_frame[f], np.int64)

    arboles = {f: cKDTree(xy[idx]) for f, idx in det_por_frame.items()}

    origen, destino, costo, sector = [], [], [], []
    for f, idx_f in sorted(det_por_frame.items()):
        for d in range(1, gap + 1):
            idx_g = det_por_frame.get(f + d)
            if idx_g is None:
                continue
            radio = vmax * d
            pares = arboles[f + d].query_ball_point(xy[idx_f], r=radio)
            for k, vecinos in enumerate(pares):
                if not vecinos:
                    continue
                i = idx_f[k]
                js = idx_g[np.array(vecinos, np.int64)]
                v = xy[js] - xy[i]
                dist = np.hypot(v[:, 0], v[:, 1])
                # La direccion de la estela dice hacia donde sigue la roca mejor
                # que dos centroides: un segmento de 40 px ya trae su angulo. Se
                # exige coherencia con la estela de ORIGEN y con la de DESTINO:
                # dos estelas de la misma roca son casi paralelas a su vuelo, y
                # pedir las dos impone suavidad sin tener que mirar el arco
                # anterior (que obligaria a un grafo de segundo orden).
                dir_link = np.arctan2(v[:, 1], v[:, 0])
                # |cos|: una estela es un segmento, no distingue ida de vuelta.
                dif_o = np.abs(np.cos(dir_link - ang[i]))
                dif_d = np.abs(np.cos(dir_link - ang[js]))

                # LA ESTELA MIDE LA VELOCIDAD. El rastro que deja una roca en un
                # frame es lo que se movio durante la exposicion, asi que el
                # salto por frame tiene que parecerse al largo de su estela. Es
                # una restriccion fisica de primer orden, y es la que separa una
                # roca (estela larga, salto grande) de una mancha de humo
                # (estela corta, salto chico) sin necesitar un grafo de segundo
                # orden que mire el arco anterior.
                lm = 0.5 * (largo[i] + largo[js])
                desaj = np.abs(dist / d - lm) / np.maximum(lm, 10.0)

                c = (w_dist * dist / (vmax * d)
                     + W_GAP * (d - 1)
                     + w_ang * (1.0 - 0.5 * (dif_o + dif_d))
                     + w_vel * np.minimum(desaj, 3.0))

                # Filtro duro de velocidad minima: por debajo de esto no es una
                # roca en vuelo, y dejarlo como penalizacion no alcanzo.
                vivo = dist >= DMIN_PX * d
                if not vivo.any():
                    continue
                # Sector de la direccion del enlace, para que el DP recuerde
                # hacia donde iba la trayectoria (ver dp_mejores).
                sect = np.floor((dir_link[vivo] + np.pi)
                                / (2 * np.pi / N_SECTORES)).astype(np.int64)
                np.clip(sect, 0, N_SECTORES - 1, out=sect)
                origen.append(np.full(int(vivo.sum()), i, np.int64))
                destino.append(js[vivo])
                costo.append(c[vivo].astype(np.float64))
                sector.append(sect)

    if not origen:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0), np.zeros(0, np.int64))
    return (np.concatenate(origen), np.concatenate(destino),
            np.concatenate(costo), np.concatenate(sector))


def dp_mejores(det, arcos, usada, tope_por_pasada, tol_sector=1,
               entrada=None):
    """Una pasada de programacion dinamica sobre el DAG temporal.

    El estado NO es solo "en que deteccion voy" sino "en que deteccion voy y en
    que direccion venia". Sin esa memoria de direccion el camino puede REBOTAR:
    como una estela es un segmento y no distingue ida de vuelta, avanzar 20 px
    al este y volver 20 px al oeste puntua igual de bien, y el DP se queda
    oscilando sobre el mismo grupo de manchas acumulando premios. Medido: 60
    trayectorias de 224 detecciones de mediana, imposibles fisicamente.

    Con el sector de llegada en el estado, una trayectoria que "va hacia el
    este" solo puede continuar hacia el este (+-`tol_sector` sectores). Es
    ademas lo que hace una roca: vuelo balistico, la direccion cambia poco.

    Devuelve hasta `tope_por_pasada` caminos disjuntos de costo negativo, del
    mejor al peor. Extraer varios por pasada en vez de uno es una aproximacion
    deliberada: recalcular el DP entero por cada trayectoria multiplica el
    tiempo por mil y cambia poco el resultado.
    """
    entrada = ENTRADA if entrada is None else entrada
    salida = entrada
    org, dst, cst, sec = arcos
    n = len(det)
    S = N_SECTORES

    # best[j, s]: mejor costo de llegar a j con el ultimo enlace en el sector s.
    best = np.full((n, S), INF)
    padre = np.full((n, S), -1, np.int64)
    padre_sec = np.full((n, S), -1, np.int8)

    ord_arco = np.argsort(det[dst, 0], kind="stable")
    org_o, dst_o, cst_o, sec_o = (org[ord_arco], dst[ord_arco],
                                  cst[ord_arco], sec[ord_arco])
    frames_dst = det[dst_o, 0]
    cortes = np.searchsorted(frames_dst, np.unique(frames_dst))
    limites = list(cortes) + [len(dst_o)]

    for a, b in zip(limites[:-1], limites[1:]):
        if a >= b:
            continue
        oi, di, ci, si = org_o[a:b], dst_o[a:b], cst_o[a:b], sec_o[a:b]
        viable = ~usada[oi] & ~usada[di]
        if not viable.any():
            continue
        oi, di, ci, si = oi[viable], di[viable], ci[viable], si[viable]

        # Abrir camino nuevo en este arco: paga ENTRADA y explica DOS detecciones.
        cand = np.full(len(oi), entrada - 2.0 * PREMIO_DET) + ci
        prev = np.full(len(oi), -1, np.int64)
        prev_s = np.full(len(oi), -1, np.int8)

        # O continuar uno que venia en un sector compatible.
        for ds in range(-tol_sector, tol_sector + 1):
            s_prev = (si + ds) % S
            llega = best[oi, s_prev]
            alt = llega + ci - PREMIO_DET
            mejor = alt < cand
            if mejor.any():
                cand[mejor] = alt[mejor]
                prev[mejor] = oi[mejor]
                prev_s[mejor] = s_prev[mejor].astype(np.int8)

        # np.minimum.at no devuelve el argmin: se ordena por (destino, sector,
        # costo) y se toma el primero de cada par (destino, sector).
        clave = di * S + si
        o = np.lexsort((cand, clave))
        clave_s, cand_s = clave[o], cand[o]
        primero = np.ones(len(o), bool)
        primero[1:] = clave_s[1:] != clave_s[:-1]
        idx = o[primero]
        d_u, s_u, c_u = di[idx], si[idx], cand[idx]
        mejora = c_u < best[d_u, s_u]
        if mejora.any():
            best[d_u[mejora], s_u[mejora]] = c_u[mejora]
            padre[d_u[mejora], s_u[mejora]] = prev[idx][mejora]
            padre_sec[d_u[mejora], s_u[mejora]] = prev_s[idx][mejora]

    total = best + salida
    total[usada, :] = INF
    plano = np.argsort(total, axis=None)

    caminos, tomadas = [], set()
    for p in plano[:tope_por_pasada * 12]:
        j, s = int(p // S), int(p % S)
        if total[j, s] >= 0:
            break
        camino, k, ks = [], j, s
        ok = True
        while k != -1:
            if k in tomadas:
                ok = False
                break
            camino.append(k)
            k_ant, s_ant = int(padre[k, ks]), int(padre_sec[k, ks])
            k, ks = k_ant, s_ant
            if k != -1 and ks < 0:
                ok = False          # cadena rota: el estado previo no existe
                break
        if not ok or len(camino) < 2:
            continue
        tomadas.update(camino)
        caminos.append((float(total[j, s]), np.array(camino[::-1], np.int64)))
        if len(caminos) >= tope_por_pasada:
            break
    return caminos


def rectitud(det, tr):
    """Desplazamiento neto / recorrido total. Es LA metrica que separa una roca
    de una culebra: un vuelo balistico da ~0.9, un camino que serpentea entre
    manchas de polvo da ~0.2, y el promedio delata al conjunto sin tener que
    mirar la imagen."""
    pts = det[tr, 1:3]
    if len(pts) < 2:
        return 0.0
    paso = np.hypot(*(pts[1:] - pts[:-1]).T).sum()
    neto = float(np.hypot(*(pts[-1] - pts[0])))
    return neto / paso if paso > 0 else 0.0


LARGA = 10          # detecciones desde las que una trayectoria "dice algo"


def resumen(det, trayectorias, usada):
    """Los numeros con los que se juzga una corrida.

    OJO con la rectitud a secas: una trayectoria de 4 detecciones es recta por
    construccion y marca 1.00, asi que ordenar por ella premia justamente a las
    configuraciones que no encontraron nada. Por eso se reporta ademas
    `rect_largas` —solo sobre las de >= LARGA detecciones— y un `score` que
    multiplica cuantas largas encontro por lo rectas que son: encontrar mucho y
    torcido puntua tan mal como encontrar poco y recto.
    """
    vacio = dict(n=0, mediana=0.0, largas=0, rectitud=0.0, rect_largas=0.0,
                 score=0.0, cobertura=0.0)
    if not trayectorias:
        return vacio
    largos = np.array([len(t) for t in trayectorias])
    rec = np.array([rectitud(det, t) for t in trayectorias])
    grandes = largos >= LARGA
    rect_largas = float(rec[grandes].mean()) if grandes.any() else 0.0
    return dict(
        n=len(trayectorias),
        mediana=float(np.median(largos)),
        largas=int(grandes.sum()),
        # media ponderada por largo: una traza de 40 puntos pesa diez veces mas
        # que una de 4, que es lo que uno quiere decir con "esto salio recto".
        rectitud=float(np.average(rec, weights=largos)),
        rect_largas=rect_largas,
        score=float(grandes.sum() * rect_largas),
        cobertura=float(usada.sum() / len(det) * 100))


def guardar(trayectorias, det, shape, usada):
    """Trayectorias en formato plano: todos los indices seguidos, mas los
    cortes. `cargar` lo devuelve como lista de arrays."""
    if trayectorias:
        plano = np.concatenate(trayectorias)
        cortes = np.cumsum([0] + [len(t) for t in trayectorias])
    else:
        plano, cortes = np.zeros(0, np.int64), np.zeros(1, np.int64)
    np.savez_compressed(TRAYECTORIAS, plano=plano, cortes=cortes,
                        det=det, shape=np.array(shape), usadas=usada)


def cargar():
    z = np.load(TRAYECTORIAS)
    plano, cortes = z["plano"], z["cortes"]
    trayectorias = [plano[a:b] for a, b in zip(cortes[:-1], cortes[1:])]
    return trayectorias, z["det"], tuple(z["shape"])


def pintar(det, trayectorias, shape, grosor=2):
    lienzo = np.zeros((shape[0], shape[1], 3), np.uint8)
    rng = np.random.default_rng(7)
    for tr in trayectorias:
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        pts = det[tr, 1:3].astype(np.int32)
        cv2.polylines(lienzo, [pts], False, color, grosor, cv2.LINE_AA)
    return lienzo


def main():
    ap = argparse.ArgumentParser(description="DETO1 etapa 2: asociacion global")
    ap.add_argument("--gap", type=int, default=5,
                    help="frames de hueco que un enlace puede saltar")
    ap.add_argument("--vmax", type=float, default=VMAX_PX)
    ap.add_argument("--min-len", type=int, default=4,
                    help="detecciones minimas para aceptar una trayectoria")
    ap.add_argument("--max-traj", type=int, default=20000)
    ap.add_argument("--por-pasada", type=int, default=400)
    # El filtro de forma vive ACA y no en la etapa 1 a proposito: la pasada por
    # el video es lo caro, asi que guarda permisivo y el recorte se ajusta
    # cuantas veces haga falta sin volver a leer el video.
    ap.add_argument("--elong", type=float, default=4.0,
                    help="largo/ancho minimo: una estela es un segmento, el "
                         "polvo es una mancha")
    ap.add_argument("--area-min", type=float, default=25.0)
    ap.add_argument("--w-dist", type=float, default=W_DIST)
    ap.add_argument("--w-ang", type=float, default=W_ANG)
    ap.add_argument("--w-vel", type=float, default=W_VEL)
    ap.add_argument("--entrada", type=float, default=ENTRADA,
                    help="costo de abrir/cerrar una trayectoria; mas bajo = "
                         "acepta trayectorias mas cortas")
    ap.add_argument("--tol-sector", type=int, default=2,
                    help="cuantos sectores de 22.5 grados puede doblar entre "
                         "enlaces consecutivos")
    ap.add_argument("--silencio", action="store_true")
    ap.add_argument("--minutos", type=float, default=0,
                    help="tope de tiempo; 0 = sin tope. Guarda lo que lleve.")
    args = ap.parse_args()

    if not DETECCIONES.exists():
        raise SystemExit(f"falta {DETECCIONES}: corre antes deto1_mascara.py")
    z = np.load(DETECCIONES)
    det, shape = z["det"], tuple(z["shape"])
    crudas = len(det)

    largo, ancho = det[:, 4], np.maximum(det[:, 5], 0.5)
    pasa = (largo / ancho >= args.elong) & (det[:, 3] >= args.area_min)
    det = det[pasa]
    print(f"{crudas} detecciones crudas -> {len(det)} tras forma "
          f"(elong>={args.elong}, area>={args.area_min:.0f})")
    if not len(det):
        raise SystemExit("el filtro de forma no dejo nada: afloja --elong")
    print(f"  frames {det[:, 0].min():.0f}-{det[:, 0].max():.0f}, "
          f"cuadro {shape[1]}x{shape[0]}, "
          f"{len(det) / max(len(np.unique(det[:, 0])), 1):.0f} por frame")

    t0 = time.time()
    print(f"construyendo arcos (gap<={args.gap}, vmax={args.vmax:.0f} px/frame)...")
    arcos = construir_arcos(det, args.gap, args.vmax,
                            args.w_dist, args.w_ang, args.w_vel)
    print(f"  {len(arcos[0]):,} arcos en {time.time() - t0:.0f}s")

    usada = np.zeros(len(det), bool)
    trayectorias = []
    pasada = 0
    while len(trayectorias) < args.max_traj:
        pasada += 1
        caminos = dp_mejores(det, arcos, usada, args.por_pasada,
                             args.tol_sector, args.entrada)
        if not caminos:
            print("  no quedan caminos de costo negativo: listo")
            break
        nuevas = 0
        for costo, camino in caminos:
            if len(camino) < args.min_len:
                usada[camino] = True      # se consume igual: es ruido corto
                continue
            usada[camino] = True
            trayectorias.append(camino)
            nuevas += 1
        if not args.silencio:
            print(f"  pasada {pasada}: +{nuevas} trayectorias "
                  f"(total {len(trayectorias)}, {usada.sum()}/{len(det)} "
                  f"detecciones usadas, {time.time() - t0:.0f}s)", flush=True)

        # Checkpoint por pasada: si esto se corta a media noche, lo encontrado
        # hasta aca sigue siendo utilizable por la mañana.
        #
        # Formato plano (indices + offsets) y no un array de objetos: las
        # trayectorias tienen largos distintos, y un `np.array(lista, dtype=
        # object)` se rompe justo cuando todas coinciden en largo (numpy lo
        # convierte en 2D sin avisar).
        guardar(trayectorias, det, shape, usada)

        if nuevas == 0:
            print("  la ultima pasada no aporto ninguna: corto")
            break
        if args.minutos and (time.time() - t0) / 60 >= args.minutos:
            print(f"  tope de {args.minutos:.0f} min alcanzado")
            break

    largos = np.array([len(t) for t in trayectorias]) if trayectorias else np.zeros(0)
    print(f"\n  {len(trayectorias)} trayectorias en {time.time() - t0:.0f}s")
    if len(largos):
        print(f"  largo: mediana {np.median(largos):.0f} detecciones, "
              f"max {largos.max()}, >=20 det: {(largos >= 20).sum()}")
        print(f"  cobertura: {usada.sum() / len(det) * 100:.1f}% de las detecciones")

    if trayectorias:
        save(OUT / "trayectorias.png", pintar(det, trayectorias, shape))
        binario = OUT / "binario_acum.png"
        if binario.exists():
            fondo = cv2.imread(str(binario), cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor((fondo // 3).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis = cv2.add(vis, pintar(det, trayectorias, shape))
            save(OUT / "trayectorias_sobre_binario.png", vis)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
