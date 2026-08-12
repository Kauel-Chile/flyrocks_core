"""
Genera los entregables a partir del JSON de trayectorias.

    uv run python debug/salidas.py [nombre_del_caso]

Entrada:  debug/casos/<caso>/entregable.json   (lo exporta la vista con el
          boton "JSON de trayectorias")
Salida:   debug/casos/<caso>/salidas/
              1_intensidad.png      la mascara sola, sin nada encima
              2_heatmap.png         de que pozos salieron mas rocas
              3_histograma.png      distribucion de alcances
              reporte.pdf           los tres, ensamblados

La imagen con las trayectorias pintadas (la que ve el cliente final) NO se
genera aca: sale de la vista con el boton "Imagen final (4K)". Es deliberado —
un solo renderizador para pantalla y para exportacion, o los dos se
desincronizan a la primera semana.

Sobre el heatmap: es RELATIVO y sin escala numerada, a proposito. La pregunta
es "de donde salio mas material", no "cuantas rocas". Ademas lo vuelve inmune a
los duplicados del detector y a que el detector mejore: si mañana encuentra un
30% mas de trayectorias, un mapa numerado cambia entero y uno relativo mantiene
la forma. Y sin numero impreso no hay numero que el cliente pueda desmentir
contando crateres.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import cv2

ROOT = Path(__file__).resolve().parent.parent
CASOS = ROOT / "debug" / "casos"

# Paleta de referencia del sistema de visualizacion, modo claro (los graficos
# van a un PDF impreso). Validada: todos los pares pasan CVD y contraste.
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIE = "#2a78d6"        # categorico slot 1 (azul)
CRITICO = "#d03b3b"      # status critical, para el umbral de evacuacion
CALIDO = "#eb6834"       # categorico slot 2 (naranja), hue del heatmap

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def log(m):
    print(f"  {m}", flush=True)


# ------------------------------------------------------------------ heatmap

def heatmap(E, caso_dir, destino):
    """Densidad relativa de rocas por pozo, sobre el terreno.

    Cada pozo aporta una gaussiana ponderada por su numero de rocas; el campo
    se normaliza a [0,1]. La rampa es de UN solo tono (naranja) con alpha
    creciente: el cero es transparente y deja ver el terreno, que es
    exactamente lo que debe hacer el valor nulo de una escala secuencial.
    """
    fondo = cv2.imread(str(caso_dir / "mascara.png"))
    if fondo is None:
        log("no encuentro mascara.png, el heatmap va sobre fondo plano")
        H, W = E["meta"]["cuadro"][1], E["meta"]["cuadro"][0]
        fondo = np.full((H, W, 3), 30, np.uint8)
    fondo = cv2.cvtColor(fondo, cv2.COLOR_BGR2RGB)
    H, W = fondo.shape[:2]

    pozos = [p for p in E["pozos"]]
    xs = np.array([p["px"][0] for p in pozos], float)
    ys = np.array([p["px"][1] for p in pozos], float)
    w = np.array([p["rocas"] for p in pozos], float)
    if w.sum() == 0:
        log("ningun pozo tiene rocas asociadas: no hay heatmap que dibujar")
        return None

    # El campo se calcula a 1/4 de resolucion: es suave por construccion, asi
    # que no se pierde nada y baja el costo a la decima parte.
    f = 4
    hh, ww = H // f, W // f
    yy, xx = np.mgrid[0:hh, 0:ww]
    campo = np.zeros((hh, ww), float)
    # Los pozos vecinos estan a ~47 px; con sigma 2.5x eso el campo se lee
    # continuo sin fundir la malla entera en una sola mancha.
    sigma = (47 * 2.5) / f
    # Peso por RAIZ del conteo, no por el conteo. La distribucion es muy
    # sesgada (aca un pozo con 81 rocas contra una mediana de 3): en escala
    # lineal ese pozo satura la rampa entero y todos los demas quedan en negro,
    # o sea el mapa deja de responder la pregunta. La raiz es la correccion
    # estandar para simbolos proporcionales — el area percibida crece con la
    # raiz — y conserva el orden, que es lo unico que este mapa promete.
    peso_vis = np.sqrt(w)
    for x, y, peso in zip(xs / f, ys / f, peso_vis):
        if peso <= 0:
            continue
        campo += peso * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    campo /= campo.max()

    campo = cv2.resize(campo, (W, H), interpolation=cv2.INTER_LINEAR)

    # Rampa de un solo tono. Alpha^0.7 para que la zona baja no desaparezca del
    # todo pero siga dejando ver el terreno.
    rampa = LinearSegmentedColormap.from_list(
        "calor", ["#4a1d00", "#8c3a06", CALIDO, "#f7a35c", "#ffe0b2"])
    rgba = rampa(campo)
    alpha = (campo ** 0.7)[..., None] * 0.85
    compuesto = (fondo / 255.0) * (1 - alpha) + rgba[..., :3] * alpha

    # Recorte a la zona util. El margen se mide contra el tamaño de la MALLA,
    # no del cuadro: relativo al cuadro 4K deja la voladura como un punto
    # rodeado de negro.
    anchoMalla = max(xs.max() - xs.min(), ys.max() - ys.min())
    m = max(0.55 * anchoMalla, 3 * sigma * f)
    x0 = max(0, int(xs.min() - m)); x1 = min(W, int(xs.max() + m))
    y0 = max(0, int(ys.min() - m)); y1 = min(H, int(ys.max() + m))
    recorte = compuesto[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(6.4, 6.4 * recorte.shape[0] / recorte.shape[1]))
    ax.imshow(np.clip(recorte, 0, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    # Los pozos, tenues, para que se lea que el campo se apoya en la malla.
    ax.scatter(xs - x0, ys - y0, s=4, c="white", alpha=0.30, linewidths=0)

    ax.set_title("¿De qué tiros salió más material?",
                 fontsize=11, weight="bold", color=INK, pad=10, loc="left")

    # Leyenda de gradiente SIN numeros: la magnitud es relativa a propósito.
    cax = fig.add_axes([0.14, 0.055, 0.30, 0.018])
    grad = np.linspace(0, 1, 256)[None, :]
    cax.imshow(grad, aspect="auto", cmap=rampa)
    cax.set_xticks([]); cax.set_yticks([]); cax.grid(False)
    for s in cax.spines.values():
        s.set_color(AXIS); s.set_linewidth(0.6)
    cax.text(-0.03, 0.5, "menos", transform=cax.transAxes, ha="right",
             va="center", fontsize=8, color=INK2)
    cax.text(1.03, 0.5, "más", transform=cax.transAxes, ha="left",
             va="center", fontsize=8, color=INK2)

    fig.text(0.14, 0.015, "Escala relativa · el color compara tiros entre sí, "
             "no cuenta rocas", fontsize=7.5, color=MUTED)
    fig.savefig(destino, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    activos = int((w > 0).sum())
    log(f"heatmap: {activos} pozos con rocas, max {int(w.max())} en un pozo")
    return destino


# --------------------------------------------------------------- histograma

def histograma(E, destino):
    """Distribucion de cuanto se alejaron las rocas del area de voladura.

    Se usa la distancia FUERA DEL AREA, no la recorrida por la roca. El radio
    de evacuacion se mide desde la voladura, no desde donde nacio cada roca:
    una que nace en el borde y viaja 80 m sale del area, y otra que nace en el
    centro y viaja los mismos 80 m puede quedarse dentro. Solo la primera
    medida es comparable con el radio, que es lo que la linea marca.
    """
    T = [t for t in E["trayectorias"] if t["estado"] == "activa"]
    d = np.array([t["salida_area_m"] or 0.0 for t in T], float)
    cens = np.array([bool(t["alcance_censurado"]) for t in T])
    radio = E["parametros"]["radio_evacuacion_m"]

    if not len(d):
        log("sin trayectorias activas: no hay histograma")
        return None

    # Bines de 10 m: mas finos fragmentan la cola, mas gruesos esconden el
    # cruce del radio de evacuacion, que es lo que hay que poder leer.
    paso = 10.0
    tope = max(radio * 1.25, float(np.ceil(d.max() / paso) * paso))
    bordes = np.arange(0, tope + paso, paso)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist([d[~cens], d[cens]], bins=bordes, stacked=True,
            color=[SERIE, SERIE], edgecolor=SURF, linewidth=0.8,
            label=["Alcance medido", "Salió del cuadro (mínimo)"])
    # La segunda serie va con textura, no con otro color: es el MISMO dato en
    # otra condicion (distancia censurada), no otra categoria.
    for p in ax.containers[1]:
        p.set_hatch("///"); p.set_facecolor("none"); p.set_edgecolor(SERIE)

    ax.axvline(radio, color=CRITICO, lw=1.8, zorder=5)
    # Las anotaciones van a media altura, no arriba: la esquina superior
    # derecha es de la leyenda y ahi se pisaban.
    tope = ax.get_ylim()[1]
    ax.annotate(f"radio de evacuación {radio:.0f} m",
                xy=(radio, tope * 0.60), xytext=(7, 0),
                textcoords="offset points", color=CRITICO, fontsize=8.5,
                weight="bold", va="center")
    ax.annotate("flyrock →", xy=(radio, tope * 0.50),
                xytext=(7, 0), textcoords="offset points",
                color=CRITICO, fontsize=8.5, va="center")

    fuera = int((d > radio).sum())
    ax.set_title("¿Hasta dónde llegó el material?",
                 fontsize=11.5, weight="bold", color=INK, loc="left", pad=18)
    # El dato principal como subtitulo: en el titulo lo dejaba ilegible de tan
    # largo, y es la cifra que se lee primero.
    ax.text(0, 1.035, f"{fuera} de {len(d)} rocas superaron el radio de evacuación",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.set_xlabel("Distancia fuera del área de voladura (m)")
    ax.set_ylabel("Rocas")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

    fig.text(0.125, -0.04,
             "Medido desde el borde del área de voladura, no desde el punto de "
             "origen de cada roca: es lo que hace comparable la distancia con "
             "el radio de evacuación.", fontsize=7.5, color=MUTED)
    fig.savefig(destino, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    log(f"histograma: {fuera}/{len(d)} sobre {radio:.0f} m, "
        f"max {d.max():.0f} m, {int(cens.sum())} censuradas")
    return destino


# ---------------------------------------------------------------------- pdf

def pdf(E, imgs, destino):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                    Table, TableStyle, KeepTogether)

    est = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=est["Heading1"], fontSize=16,
                        textColor=colors.HexColor(INK), spaceAfter=2)
    sub = ParagraphStyle("sub", parent=est["Normal"], fontSize=9,
                         textColor=colors.HexColor(MUTED), spaceAfter=14)
    h2 = ParagraphStyle("h2", parent=est["Heading2"], fontSize=11,
                        textColor=colors.HexColor(INK), spaceBefore=10, spaceAfter=6)
    nota = ParagraphStyle("nota", parent=est["Normal"], fontSize=8,
                          textColor=colors.HexColor(MUTED), spaceBefore=4)

    R, P = E["resumen"], E["parametros"]
    doc = SimpleDocTemplate(str(destino), pagesize=letter, topMargin=54,
                            bottomMargin=48, leftMargin=48, rightMargin=48)
    S = []
    S.append(Paragraph("Análisis de proyecciones — Flyrocks", h1))
    S.append(Paragraph(
        f"Caso {E['meta']['caso']} · {E['meta']['video']} · generado {E['meta']['generado']}", sub))

    niveles = R["por_nivel_de_origen"]
    filas = [
        ["Trayectorias analizadas", f"{R['activas']}"],
        ["Tiros de origen identificados", f"{niveles.get('pozo', 0)}"],
        ["Origen acotado a un grupo de tiros", f"{niveles.get('grupo', 0)}"],
        ["Origen acotado a un sector", f"{niveles.get('sector', 0)}"],
        ["Tiros que proyectaron material", f"{R['pozos_con_rocas']}"],
        [f"Rocas fuera del radio de evacuación ({P['radio_evacuacion_m']:.0f} m)",
         f"{R['fuera_del_radio_evacuacion']}"],
        ["Alcance máximo registrado", f"{R['alcance_max_m']:.0f} m"],
        ["Alcance mediano", f"{R['alcance_mediano_m']:.0f} m"],
    ]
    t = Table(filas, colWidths=[4.3*inch, 1.3*inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor(INK2)),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor(INK)),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(GRID)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    S.append(t)

    # KeepTogether para que un titulo no quede huerfano al pie de pagina con su
    # figura al otro lado del salto.
    if imgs.get("heatmap"):
        S.append(KeepTogether([
            Paragraph("Origen del material proyectado", h2),
            Image(str(imgs["heatmap"]), width=5.0*inch, height=5.0*inch,
                  kind="proportional"),
            Paragraph("Escala relativa: el color compara los tiros entre sí. "
                      "No representa un conteo de rocas.", nota),
        ]))

    if imgs.get("histograma"):
        S.append(KeepTogether([
            Paragraph("Distribución de alcances", h2),
            Image(str(imgs["histograma"]), width=6.4*inch, height=3.0*inch,
                  kind="proportional"),
        ]))

    S.append(Paragraph("Cómo leer este reporte", h2))
    ct = P["calce_temporal"]
    temporal = (f"activo (causalidad ±{ct['tolerancia_causal_frames']} frames, "
                f"retardo {ct['retardo_frames']} frames)") if ct["activo"] else "no aplicado"
    S.append(Paragraph(
        "El tiro de origen de cada trayectoria se obtiene prolongando su tangente "
        "inicial hacia atrás y cruzando el resultado con los tiempos de la secuencia "
        "de detonación. El sistema entrega <b>candidatos con su confianza</b>, no un "
        "tiro único: cuando la geometría y el tiempo no bastan para separar dos tiros "
        "vecinos, el origen se reporta como grupo o como sector. "
        f"Calce temporal: {temporal}. Corrección de paralaje k = {P['paralaje_k']}. "
        f"Apertura de búsqueda σ = {P['apertura_sigma_grados']}°. "
        f"Homografía con error RMS de {P['homografia']['rms_px']} px.", nota))
    if P["paralaje_k"] == 0:
        S.append(Paragraph(
            "<b>Nota técnica:</b> la corrección de paralaje está desactivada (k = 0). "
            "El punto donde una roca se hace visible no coincide con la boca del tiro, "
            "así que el origen estimado tiene un sesgo sistemático hacia afuera.", nota))

    doc.build(S)
    log(f"pdf: {destino.name} ({destino.stat().st_size / 1024:.0f} KB)")
    return destino


# --------------------------------------------------------------------- main

def main():
    nombre = sys.argv[1] if len(sys.argv) > 1 else "3160-789"
    caso_dir = CASOS / nombre
    fuente = caso_dir / "entregable.json"
    if not fuente.exists():
        raise SystemExit(
            f"falta {fuente}\n"
            f"  Expórtalo desde la vista con el botón «JSON de trayectorias».")

    E = json.loads(fuente.read_text(encoding="utf-8"))
    out = caso_dir / "salidas"
    out.mkdir(exist_ok=True)
    print(f"\n=== SALIDAS: {nombre}  ->  {out.relative_to(ROOT)}\n")

    R = E["resumen"]
    log(f"{R['activas']} trayectorias activas | {R['pozos_con_rocas']} tiros con material")

    # 1 — la mascara sola, sin nada encima
    src = caso_dir / "mascara.png"
    if src.exists():
        dst = out / "1_intensidad.png"
        dst.write_bytes(src.read_bytes())
        log(f"1_intensidad.png ({dst.stat().st_size / 1e6:.1f} MB)")

    imgs = {}
    imgs["heatmap"] = heatmap(E, caso_dir, out / "2_heatmap.png")
    imgs["histograma"] = histograma(E, out / "3_histograma.png")
    pdf(E, imgs, out / "reporte.pdf")

    print("\n=== listo.\n")


if __name__ == "__main__":
    main()
