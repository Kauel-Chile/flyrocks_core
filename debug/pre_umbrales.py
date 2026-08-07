"""
PASO 6 — Contact sheet de umbrales (acotar el rango antes de invertir en la UI).

Compara TRES formas de binarizar, sobre la mascara de LINEALIDAD:
  A) umbral simple           -> es lo mismo que el percentil de energia YA
                                DESCARTADO por matar rocas tenues. Se incluye
                                solo para tener la evidencia a la vista.
  B) HISTERESIS (2 umbrales) -> alto = semillas confiables; bajo = crecer desde
                                las semillas por conectividad. Una traza tenue
                                que TOCA una semilla sobrevive completa; el ruido
                                tenue AISLADO muere.
  C) histeresis + filtro de forma -> ademas descarta componentes poco alargados
                                (relacion de aspecto), que son manchas de humo.

Salidas en 7_preproceso/04_binaria/.

    uv run python debug/pre_umbrales.py
"""
import cv2
import numpy as np

from pre_common import (D_LINEAL, D_BINARIA, CACHE, ensure_dirs, to8, grid,
                        label, save)
import pre_pasada
import pre_render


def hysteresis(img, lo, hi):
    """Binariza por doble umbral: conserva las componentes de (img>lo) que
    contienen al menos un pixel de (img>hi)."""
    weak = (img > lo).astype(np.uint8)
    strong = img > hi
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    if n <= 1:
        return np.zeros_like(weak, bool)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False                                  # fondo
    return keep[lab]


def hysteresis_guarded(img, lo, hi, max_area=4000):
    """Histeresis CON SALVAGUARDA contra derrame.

    La histeresis pura se derrama: en la zona del blast la densidad es tal que
    el umbral bajo conecta todo en un unico blob gigante (verificado: 83
    componentes cubriendo 4.2% del cuadro). Aqui, por cada componente que
    contiene semilla:
      - si es chica (area <= max_area) se acepta ENTERA  -> la traza tenue
        aislada sobrevive completa, que es el objetivo de la histeresis;
      - si es un derrame (area > max_area) se conserva solo su nucleo fuerte
        -> el blob del blast se reduce a lo que el umbral alto ya veia.
    """
    weak = (img > lo).astype(np.uint8)
    strong = img > hi
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    if n <= 1:
        return np.zeros_like(weak, bool)
    has_seed = np.zeros(n, bool)
    has_seed[np.unique(lab[strong])] = True
    has_seed[0] = False
    areas = stats[:, cv2.CC_STAT_AREA]
    grow = has_seed & (areas <= max_area)          # se aceptan enteras
    out = grow[lab]
    spill = has_seed & (areas > max_area)          # solo su nucleo fuerte
    out |= spill[lab] & strong
    return out


def shape_filter(mask, min_area=40, min_aspect=2.5):
    """Descarta componentes poco alargadas (manchas) via elipse ajustada."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, bool)
    kept = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # aspecto del bounding box: barato y suficiente para descartar blobs
        asp = max(w, h) / max(1.0, min(w, h))
        if asp < min_aspect:
            continue
        out |= (lab == i)
        kept += 1
    return out, kept


def length_filter(mask, min_len):
    """Descarta componentes CORTAS.

    En la zona del blast quedan muchos fragmentos cortos (ruido de la explosion)
    que el realce de linealidad enciende igual que una traza real: localmente son
    lineas. Lo que los separa de una trayectoria no es el brillo ni la forma
    local, es el LARGO. Se mide como la diagonal del bounding box (barato; para
    curvas subestima algo, lo que juega a favor: es conservador).
    """
    if min_len <= 0:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    w = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    keep = np.hypot(w, h) >= min_len
    keep[0] = False
    return keep[lab]


def ncomp(mask):
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return n - 1


def main():
    ensure_dirs()
    lin = np.load(CACHE / "linealidad.npz")["lin_ff"]
    acc_ff = pre_pasada.load()["acc_ff"]
    box = pre_render.auto_crop(acc_ff, 24)
    cut = lambda im: pre_render.cut(im, box)          # noqa: E731

    pcts = [90, 95, 97, 98, 99, 99.5]
    thr = {p: float(np.percentile(lin, p)) for p in pcts}

    # ---------- A) umbral simple --------------------------------------
    print("A) UMBRAL SIMPLE sobre linealidad")
    items = []
    for p in pcts:
        m = lin > thr[p]
        print(f"    p{p:<5} (>{thr[p]:5.1f})  {100*m.mean():5.2f}% del cuadro  "
              f"{ncomp(m):>7,} componentes")
        items.append(((cut(m) * 255).astype(np.uint8),
                      f"simple p{p} (>{thr[p]:.1f})"))
    save(D_BINARIA / "A_umbral_simple.png", grid(items, cols=3))

    # ---------- B) histeresis -----------------------------------------
    print("\nB) HISTERESIS (semilla alta + crecimiento bajo)")
    items = []
    pares = [(90, 98), (90, 99), (95, 99), (95, 99.5), (97, 99.5), (98, 99.9)]
    best = None
    for plo, phi in pares:
        lo = float(np.percentile(lin, plo))
        hi = float(np.percentile(lin, phi))
        m = hysteresis(lin, lo, hi)
        nc = ncomp(m)
        print(f"    lo=p{plo} hi=p{phi}   {100*m.mean():5.2f}% del cuadro  "
              f"{nc:>7,} componentes")
        items.append(((cut(m) * 255).astype(np.uint8),
                      f"hist lo=p{plo} hi=p{phi}"))
        if plo == 95 and phi == 99:
            best = m
    save(D_BINARIA / "B_histeresis.png", grid(items, cols=3))

    # ---------- C) histeresis CON SALVAGUARDA --------------------------
    print("\nC) HISTERESIS CON SALVAGUARDA (limite de derrame por componente)")
    items = []
    for ma in (1000, 4000, 15000):
        m = hysteresis_guarded(lin, thr[95], thr[99], max_area=ma)
        print(f"    max_area={ma:>6}:  {100*m.mean():5.2f}% del cuadro  "
              f"{ncomp(m):>7,} componentes")
        items.append(((cut(m) * 255).astype(np.uint8),
                      f"guardada p95/p99 max_area={ma}"))
    items.append(((cut(best) * 255).astype(np.uint8),
                  "histeresis SIN salvaguarda (se derrama)"))
    save(D_BINARIA / "C_histeresis_salvaguarda.png", grid(items, cols=2))

    # ---------- el contraste que importa ------------------------------
    lin8 = to8(lin, hi_pct=99.8)
    m_simple = lin > thr[99]
    m_hist = hysteresis(lin, thr[95], thr[99])
    m_guard = hysteresis_guarded(lin, thr[95], thr[99], max_area=4000)
    save(D_BINARIA / "comparacion_simple_vs_histeresis.png", grid([
        (cut(lin8), "linealidad (gris, se conserva)"),
        ((cut(m_simple) * 255).astype(np.uint8),
         "UMBRAL SIMPLE p99 (limpio, pero corta lo tenue)"),
        ((cut(m_hist) * 255).astype(np.uint8),
         "HISTERESIS pura p95/p99 (SE DERRAMA en el blast)"),
        ((cut(m_guard) * 255).astype(np.uint8),
         "HISTERESIS CON SALVAGUARDA (lo mejor de ambas)"),
    ], cols=2))
    # ---------- D) limpiar los fragmentos cortos del blast -------------
    print("\nD) FILTRO DE LARGO (los fragmentos cortos del centro)")
    items = [((cut(m_guard) * 255).astype(np.uint8), "sin filtro de largo")]
    for ml in (40, 80, 150, 250):
        m = length_filter(m_guard, ml)
        print(f"    largo minimo {ml:>3} px:  {ncomp(m):>6,} componentes  "
              f"({100*m.mean():5.2f}% del cuadro)")
        items.append(((cut(m) * 255).astype(np.uint8),
                      f"largo minimo = {ml} px"))
    save(D_BINARIA / "D_filtro_largo.png", grid(items, cols=3))

    # ---------- versiones de CUADRO COMPLETO (sin recorte) -------------
    # Los mosaicos de arriba usan un recorte para que se aprecie el detalle;
    # estas son para revisar la escena entera.
    print("\nCUADRO COMPLETO (sin recorte)")
    m80 = length_filter(m_guard, 80)
    save(D_BINARIA / "FULL_sin_filtro.png", (m_guard * 255).astype(np.uint8))
    save(D_BINARIA / "FULL_largo80.png", (m80 * 255).astype(np.uint8))
    save(D_BINARIA / "FULL_linealidad_gris.png", to8(lin, hi_pct=99.8))
    save(D_BINARIA / "FULL_comparacion_largo.png", grid([
        ((m_guard * 255).astype(np.uint8), "sin filtro de largo"),
        ((m80 * 255).astype(np.uint8), "largo minimo = 80 px"),
    ], cols=1, max_w=3800))
    # el binario sobre el gris: para ver QUE se conservo y que se boto
    ov = cv2.cvtColor((to8(lin, hi_pct=99.8) * 0.4).astype(np.uint8),
                      cv2.COLOR_GRAY2BGR)
    ov[m_guard & ~m80] = (60, 60, 255)      # rojo  = descartado por corto
    ov[m80] = (60, 255, 60)                 # verde = conservado
    save(D_BINARIA / "FULL_overlay_que_se_boto.png",
         label(ov, "verde = se conserva (>80px)   rojo = descartado por corto"))

    print(f"\n  simple p99       -> {ncomp(m_simple):>6,} componentes  "
          f"({100*m_simple.mean():.2f}% del cuadro)")
    print(f"  histeresis pura  -> {ncomp(m_hist):>6,} componentes  "
          f"({100*m_hist.mean():.2f}% del cuadro)  <- derrame")
    print(f"  con salvaguarda  -> {ncomp(m_guard):>6,} componentes  "
          f"({100*m_guard.mean():.2f}% del cuadro)")


if __name__ == "__main__":
    main()
