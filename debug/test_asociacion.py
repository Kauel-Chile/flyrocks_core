"""
Banco de pruebas de la asociacion trayectoria -> pozo de origen.

Replica el algoritmo de demo/trayectorias.html (funcion `asociar`) y lo corre
contra trayectorias SINTETICAS con origen conocido, para saber que tasa de
acierto esperar ANTES de ponerse a dibujar a mano.

El experimento: por cada uno de los 113 pozos se simula una roca que sale en
direccion radial desde el centro del blast, se hace visible unos metros despues
(el humo y el destello tapan el arranque) y se dibuja con cierto error de
angulo. Se mide si el pozo verdadero queda 1o, entre los 3 primeros, o afuera.

Los dos parametros que mueven la aguja:
    visible_m   a que distancia del pozo aparece el trazo (cuanto tapa el humo)
    err_grados  cuanto se equivoca el trazo a mano en la direccion inicial

    uv run python debug/test_asociacion.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MALLA = ROOT / "debug" / "demo" / "malla.json"

SIGMA = 4.0       # apertura de la cuna, en grados (valor por defecto del visor)
K = 0.04          # paralaje
RNG = np.random.default_rng(7)


def cargar():
    d = json.loads(MALLA.read_text(encoding="utf-8"))
    P = np.array([[p["px"], p["py"]] for p in d["pozos"]])
    ids = [p["id"] for p in d["pozos"]]
    A = np.array(d["meta"]["A"])
    pxm = np.sqrt(abs(np.linalg.det(A)))
    return P, ids, pxm, d


def asociar(p0, c1, P, pxm, nadir, k=None, sigma=None):
    """Puerto directo de la funcion `asociar` del visor. Devuelve indices
    ordenados por costo.

    k y sigma se resuelven EN LA LLAMADA, no en la firma: los defaults de
    Python se evaluan al definir la funcion, asi que ponerlos ahi hacia que
    reasignar los globales del modulo no tuviera ningun efecto (y un barrido
    de parametros saliera plano por construccion).
    """
    k = K if k is None else k
    sigma = SIGMA if sigma is None else sigma
    f = 1 + k
    O = nadir + (p0 - nadir) / f

    d = p0 - c1
    n = np.hypot(*d)
    if n < 1e-6:
        return np.array([]), np.array([])
    d = d / n

    tan = np.tan(np.radians(sigma))
    a_min, a_ref, a_tol = 15 * pxm, 70 * pxm, -8 * pxm

    V = P - O
    a = V @ d
    perp = np.abs(V[:, 0] * d[1] - V[:, 1] * d[0])

    ang = perp / (tan * np.maximum(a, a_min))
    lej = np.maximum(0, a) / a_ref
    costo = ang ** 2 + lej ** 2
    costo = np.where(a < a_tol, np.inf, costo)

    orden = np.argsort(costo)
    orden = orden[np.isfinite(costo[orden])]
    return orden, costo


def corrida(P, pxm, nadir, visible_m, err_grados, largo_m=120, n_rep=6,
            k=None, sigma=None, k_real=0.0):
    """Por cada pozo, n_rep trayectorias sinteticas. Devuelve (top1, top3).

    `k_real` es el paralaje con el que se GENERA (la roca ya subio algo cuando
    se hace visible, y eso corre su imagen hacia afuera del nadir); `k` es el
    que se ASUME al asociar. Separarlos es lo unico que permite medir si
    corregir el paralaje ayuda: con k_real=0 el test premia no corregir por
    construccion, que no dice nada.
    """
    centro = P.mean(0)
    top1 = top3 = tot = 0
    for i in range(len(P)):
        base = P[i] - centro
        if np.hypot(*base) < 1e-6:
            base = np.array([1.0, 0.0])
        base = base / np.hypot(*base)          # direccion radial desde el blast
        for _ in range(n_rep):
            th = np.radians(RNG.normal(0, err_grados))
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            dirv = R @ base
            # el trazo arranca donde la roca se hace visible, no en el pozo
            p0 = P[i] + dirv * visible_m * pxm
            c1 = p0 + dirv * largo_m * pxm / 2
            # ...y lo que se VE esta corrido hacia afuera del nadir
            p0 = nadir + (p0 - nadir) * (1 + k_real)
            c1 = nadir + (c1 - nadir) * (1 + k_real)
            orden, _ = asociar(p0, c1, P, pxm, nadir, k, sigma)
            if len(orden):
                if orden[0] == i: top1 += 1
                if i in orden[:3]: top3 += 1
            tot += 1
    return 100*top1/tot, 100*top3/tot


def main():
    P, ids, pxm, d = cargar()
    nadir = np.array([d["meta"]["ancho"] / 2, d["meta"]["alto"] / 2])
    print(f"113 pozos · {pxm:.2f} px/m · nadir {nadir.astype(int)} · "
          f"cuna {SIGMA}° · k={K}\n")

    vis = [5, 10, 20, 40]
    err = [0, 2, 5, 10]

    for etiqueta, idx in (("ACIERTO EXACTO (top-1)", 0), ("ENTRE LOS 3 (top-3)", 1)):
        print(etiqueta)
        print("            " + "".join(f"{e:>4}° err" for e in err))
        for v in vis:
            fila = f"  visible {v:>2} m ".ljust(12)
            for e in err:
                fila += f"{corrida(P, pxm, nadir, v, e)[idx]:>8.0f}%"
            print(fila)
        print()

    print("Lectura: 'visible' = a cuantos metros del pozo arranca el trazo "
          "(cuanto tapa el humo);\n'err' = error de angulo del trazo a mano.")


if __name__ == "__main__":
    main()
