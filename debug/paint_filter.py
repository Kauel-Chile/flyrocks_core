"""
paint_filter.py  —  Filtro por MASCARA PINTADA (3 colores) para el pipeline.

Toma un PNG pintado a mano SOBRE la imagen de trabajo (misma resolucion y
encuadre que el video) y lo convierte en un filtro espacial (ROI) para el
pipeline de deteccion de flyrocks.

--------------------------------------------------------------------------
SEMANTICA DE COLORES  (cada color es una CLASE/intencion, NO una identidad;
por eso un color por clase basta y los cruces no importan)
--------------------------------------------------------------------------
  VERDE  (0,255,0) = humo / no interes      -> se DESCARTA
  ROJO   (255,0,0) = zona con flyrocks      -> se EXTRAE (aunque este sobre humo)
  AZUL   (0,0,255) = critico / no perder    -> se EXTRAE + relajar filtros aguas
                                               abajo (garantizar captura)

Regla de extraccion recomendada (a nivel de punto x,y):
    conservar  =  (ROJO o AZUL)  y  (no VERDE)
Todo lo NO pintado (negro/gris) queda FUERA del ROI (no se extrae).

--------------------------------------------------------------------------
COMO SE USA
--------------------------------------------------------------------------
Se aplica a PUNTOS (x, y): sirve igual para eventos crudos, centroides de
deteccion, o puntos de trayectoria. Solo depende de numpy + OpenCV.

    pm   = PaintMask("pintado.png", work_w=W, work_h=H)  # W,H = resolucion de trabajo
    keep = pm.keep(xs, ys)          # bool: que puntos extraer (dentro del ROI)
    xs, ys = xs[keep], ys[keep]
    crit = pm.in_blue(xs, ys)       # bool: cuales son criticos (relajar filtros)

Si tu pipeline trabaja a OTRA resolucion que el PNG pintado, pasa work_w/work_h
y la mascara se redimensiona sola (vecino-mas-cercano, sin mezclar colores).
"""
import numpy as np
import cv2


class PaintMask:
    """Parsea el PNG pintado en 3 mascaras booleanas y responde por punto."""

    def __init__(self, png_path, work_w=None, work_h=None):
        img = cv2.imread(str(png_path))                      # BGR
        if img is None:
            raise FileNotFoundError(f"No pude leer el PNG: {png_path}")
        # si el pipeline trabaja a otra resolucion, ajusta la mascara
        if work_w and work_h and (img.shape[1] != work_w or img.shape[0] != work_h):
            img = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_NEAREST)

        b = img[:, :, 0].astype(int)
        g = img[:, :, 1].astype(int)
        r = img[:, :, 2].astype(int)
        # "un canal domina claramente" -> robusto a compresion / antialias de bordes
        self.red   = (r > 100) & (r > g + 50) & (r > b + 50)
        self.green = (g > 100) & (g > r + 50) & (g > b + 50)
        self.blue  = (b > 100) & (b > r + 50) & (b > g + 50)
        self.h, self.w = img.shape[:2]

    def _at(self, mask, xs, ys):
        xs = np.clip(np.asarray(xs).astype(int), 0, self.w - 1)
        ys = np.clip(np.asarray(ys).astype(int), 0, self.h - 1)
        return mask[ys, xs]

    def keep(self, xs, ys):
        """ROI de extraccion: (rojo o azul) y no verde. -> bool por punto."""
        return (self._at(self.red, xs, ys) | self._at(self.blue, xs, ys)) \
            & ~self._at(self.green, xs, ys)

    def in_blue(self, xs, ys):
        """Criticos: aqui conviene relajar rectitud/largo/energia y garantizar."""
        return self._at(self.blue, xs, ys)

    def in_green(self, xs, ys):
        return self._at(self.green, xs, ys)

    def in_red(self, xs, ys):
        return self._at(self.red, xs, ys)


# ==========================================================================
# OPCIONAL: mismo filtro como NODO del pipeline original (arquitectura
# PipelineNode con run(context)). Se enchufa JUSTO DESPUES del extractor de
# eventos y ANTES de DBSCAN.
# ==========================================================================
class PaintROINode:
    def __init__(self, name, png_path, work_w, work_h):
        self.name = name
        self.pm = PaintMask(png_path, work_w, work_h)

    def run(self, context):
        tensor = context.get("tensor_raw")               # [x, y, t, intensidad]
        if tensor is None:
            return context
        keep = self.pm.keep(tensor[:, 0], tensor[:, 1])
        context["tensor_raw"] = tensor[keep]             # solo eventos del ROI
        context["paint_mask"] = self.pm                  # para el azul, mas adelante
        return context


# ==========================================================================
# EJEMPLO de integracion en el PIPELINE ORIGINAL
# ==========================================================================
if __name__ == "__main__":
    # --- 1) a nivel de EVENTOS (justo despues del EventExtractor) ---
    #   tensor = context["tensor_raw"]            # [x, y, t, intensidad]
    #   pm = PaintMask("pintado.png", work_w=3840, work_h=2160)
    #   tensor = tensor[pm.keep(tensor[:, 0], tensor[:, 1])]
    #   context["tensor_raw"] = tensor
    #   ...luego DBSCAN / Kalman / etc. igual que siempre...
    #
    # --- 2) usar el AZUL al filtrar trayectorias ---
    #   for traj in trayectorias:                 # traj: array de puntos (x, y)
    #       critico = pm.in_blue(traj[:, 0], traj[:, 1]).any()
    #       umbral  = UMBRAL_RELAJADO if critico else UMBRAL_NORMAL
    #       if metrica(traj) >= umbral:
    #           conservar(traj)
    #
    # --- 3) como NODO en la cadena original ---
    #   pipeline = extractor | PaintROINode("ROI", "pintado.png", 3840, 2160) \
    #              | dbscan | tracker | ...
    print("paint_filter: importa PaintMask (o PaintROINode) y usa keep()/in_blue().")
