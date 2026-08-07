"""
PASO 7 — Herramienta interactiva de binarizacion (doble slider + salvaguarda).

Calibras mirando el ACUMULADO (izquierda), que es donde se ven las estelas, y
al lado ves como queda un FRAME SUELTO con ese mismo corte (derecha) — para no
llevarse la sorpresa de un acumulado bonito con frames individuales vacios.

La mascara gris NUNCA se pierde: lo que exporta esta herramienta es una CAPA
BINARIA adicional + el JSON con los parametros usados.

Controles, EN EL ORDEN EN QUE SE USAN:
  1.FUENTE      SELECTOR (no es un rango): 0=linealidad 1=intensidad 2=z-score.
                Se ve UNA a la vez. De esta eleccion depende todo lo demas.
  2.ALTO hi     percentil del umbral ALTO = que es lo bastante evidente como
                para DECLARAR que ahi hay una trayectoria (semillas).
  3.BAJO lo     percentil del umbral BAJO = hasta donde se sigue dibujando una
                trayectoria ya declarada (crecimiento). No crea trazas nuevas.
  4.max_area    freno anti-derrame; 0 = histeresis pura (se derrama en el blast)
  5.largo_min   descarta trazas mas cortas que X px (limpia el ruido del centro)
  6.frame       cual frame suelto se muestra a la derecha (no afecta la mascara)

Teclas:
  1 2 3   cambiar de fuente directamente
  g       exportar la mascara en RESOLUCION COMPLETA + JSON de parametros
  q       salir

    uv run python debug/pre_slider.py            # interactivo
    uv run python debug/pre_slider.py --test     # sin ventana (verifica que corre)
"""
import argparse
import json

import cv2
import numpy as np

from pre_common import (CACHE, D_BINARIA, ensure_dirs, to8, label, save)
from pre_umbrales import hysteresis_guarded, length_filter, ncomp
import pre_pasada
import pre_linealidad

VIEW_SCALE = 0.30          # la vista interactiva va reducida (fluidez en 4K)
WIN = "binarizacion — acumulado (izq)  vs  frame suelto (der)"


def build_sources(view=True):
    """Devuelve las tres fuentes (acumulado) y los frames sueltos, ya reducidos
    si view=True. La linealidad por-frame se cachea (es cara)."""
    d = pre_pasada.load()
    lin = np.load(CACHE / "linealidad.npz")["lin_ff"]

    acc = {
        "linealidad": lin,
        "intensidad": d["acc_ff"].astype(np.float32),
        "z-score": d["acc_zff"].astype(np.float32),
    }

    ids = [int(i) for i in d["sample_ids"]]
    fcache = CACHE / "frames_linealidad.npz"
    if fcache.exists():
        fz = np.load(fcache)
        frames = {k: fz[k].astype(np.float32) for k in fz.files}
    else:
        print("Precomputando linealidad de los frames de muestra (una vez)...")
        frames = {}
        for i in ids:
            ff = d[f"ff_{i}"].astype(np.float32)
            li, _, _ = pre_linealidad.lineness(ff, 51, 16, mode="disco")
            frames[f"lin_{i}"] = li
            frames[f"int_{i}"] = ff
            frames[f"z_{i}"] = d[f"zff_{i}"].astype(np.float32)
            print(f"  frame {i} listo")
        np.savez_compressed(fcache, **frames)
    if view:
        rs = lambda a: cv2.resize(a, None, fx=VIEW_SCALE, fy=VIEW_SCALE,   # noqa: E731
                                  interpolation=cv2.INTER_AREA)
        acc = {k: rs(v) for k, v in acc.items()}
        frames = {k: rs(v) for k, v in frames.items()}
    return acc, frames, ids


def overlay(gray_src, mask):
    """Binario en verde sobre el gris atenuado (da contexto de donde cae)."""
    g = to8(gray_src, hi_pct=99.8)
    out = cv2.cvtColor((g * 0.35).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out[mask] = (60, 255, 60)
    return out


def compute(src, plo, phi, max_area):
    lo = float(np.percentile(src, plo))
    hi = float(np.percentile(src, phi))
    if max_area <= 0:
        from pre_umbrales import hysteresis
        return hysteresis(src, lo, hi), lo, hi
    return hysteresis_guarded(src, lo, hi, max_area), lo, hi


def main(test=False):
    ensure_dirs()
    acc, frames, ids = build_sources(view=True)
    names = ["linealidad", "intensidad", "z-score"]
    pref = {"linealidad": "lin", "intensidad": "int", "z-score": "z"}

    state = dict(lo=950, hi=990, area=4000, largo=0, src=0, fr=0)

    def render():
        name = names[state["src"]]
        plo = min(state["lo"] / 10.0, 99.8)
        phi = min(max(state["hi"] / 10.0, plo + 0.1), 99.95)
        # el slider va en px de resolucion COMPLETA; la vista esta reducida
        largo_view = state["largo"] * VIEW_SCALE
        A = acc[name]
        mA, lo, hi = compute(A, plo, phi, state["area"])
        mA = length_filter(mA, largo_view)
        fid = ids[state["fr"] % len(ids)]
        F = frames[f"{pref[name]}_{fid}"]
        # el frame suelto se corta con los MISMOS valores absolutos, no con su
        # propio percentil: asi la comparacion es honesta
        if state["area"] <= 0:
            from pre_umbrales import hysteresis
            mF = hysteresis(F, lo, hi)
        else:
            mF = hysteresis_guarded(F, lo, hi, state["area"])
        mF = length_filter(mF, largo_view)

        left = overlay(A, mA)
        left = label(left, f"FUENTE ACTIVA: {name.upper()}", (14, 34), 0.9,
                     color=(60, 255, 255))
        left = label(left, f"lo=p{plo:.1f} ({lo:.1f})  hi=p{phi:.1f} ({hi:.1f})"
                           f"  max_area={state['area']}"
                           f"  largo_min={state['largo']}px", (14, 70), 0.6)
        left = label(left, f"{100*mA.mean():.2f}% del cuadro   "
                           f"{ncomp(mA):,} componentes", (14, 102), 0.6)

        right = overlay(F, mF)
        right = label(right, f"FRAME SUELTO #{fid}  (mismos umbrales)",
                      (14, 34), 0.8)
        right = label(right, f"{100*mF.mean():.3f}% del cuadro   "
                             f"{ncomp(mF):,} componentes", (14, 70), 0.6)

        sep = np.full((left.shape[0], 6, 3), 200, np.uint8)
        return np.hstack([left, sep, right]), (name, plo, phi, lo, hi)

    if test:
        img, meta = render()
        save(D_BINARIA / "slider_preview.png", img)
        print(f"  preview OK con {meta}")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1800, 700)
    # ORDEN DELIBERADO: primero se elige el algoritmo (de el depende todo lo
    # demas), luego ALTO y BAJO en el orden en que actuan, luego los ajustes.
    cv2.createTrackbar("1.FUENTE 0lin 1int 2zsc", WIN, state["src"], 2,
                       lambda v: state.__setitem__("src", v))
    cv2.createTrackbar("2.ALTO  hi x10", WIN, state["hi"], 999,
                       lambda v: state.__setitem__("hi", v))
    cv2.createTrackbar("3.BAJO  lo x10", WIN, state["lo"], 998,
                       lambda v: state.__setitem__("lo", v))
    cv2.createTrackbar("4.max_area", WIN, state["area"], 30000,
                       lambda v: state.__setitem__("area", v))
    cv2.createTrackbar("5.largo_min px", WIN, state["largo"], 600,
                       lambda v: state.__setitem__("largo", v))
    cv2.createTrackbar("6.frame muestra", WIN, state["fr"], len(ids) - 1,
                       lambda v: state.__setitem__("fr", v))

    print(__doc__)
    while True:
        img, (name, plo, phi, lo, hi) = render()
        cv2.imshow(WIN, img)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("1"), ord("2"), ord("3")):
            state["src"] = k - ord("1")
            cv2.setTrackbarPos("1.FUENTE 0lin 1int 2zsc", WIN, state["src"])
        if k == ord("g"):
            print("Exportando en resolucion completa...")
            full_acc, _, _ = build_sources(view=False)
            src = full_acc[name]
            m, lo_f, hi_f = compute(src, plo, phi, state["area"])
            m = length_filter(m, state["largo"])       # aqui SI en px full-res
            save(D_BINARIA / "mascara_binaria_export.png",
                 (m * 255).astype(np.uint8))
            params = dict(fuente=name, percentil_lo=plo, percentil_hi=phi,
                          umbral_lo=lo_f, umbral_hi=hi_f,
                          max_area=state["area"], largo_min_px=state["largo"],
                          pct_cuadro=float(100 * m.mean()),
                          componentes=int(ncomp(m)),
                          nota="capa binaria adicional; la mascara gris se conserva")
            (D_BINARIA / "mascara_binaria_params.json").write_text(
                json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {params}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args().test)
