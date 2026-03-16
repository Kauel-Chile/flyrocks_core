import cv2
import numpy as np
import json

class ResultsExporter:
    def __init__(self, config):
        """
        config debe tener:
        - H_MATRIX: La matriz de 3x3 que sale del FlyRock Adjuster.
        - EXPECTED_PROJECTION_ZONE: Polígono de seguridad.
        - EDGE_MARGIN: Margen de la imagen.
        """
        self.config = config
        # Pre-calculamos la inversa para mayor eficiencia
        try:
            self.H_inv = np.linalg.inv(self.config.H_MATRIX)
        except np.linalg.LinAlgError:
            print("Error: La matriz H no es invertible.")
            self.H_inv = np.eye(3)

    def apply_homography(self, pt):
        """
        Transforma un punto de la IMAGEN al MUNDO (metros).
        Usa coordenadas homogéneas: [x, y, 1] -> [X, Y, Z] -> [X/Z, Y/Z]
        """
        x, y = pt
        # Crear vector de punto en la imagen
        p = np.array([float(x), float(y), 1.0], dtype=np.float64)
        
        # Multiplicar por la inversa (Imagen -> Mundo)
        P = np.dot(self.H_inv, p)
        
        # División por Z para normalizar la proyección
        if abs(P[2]) > 1e-10:
            return P[0] / P[2], P[1] / P[2]
        return 0.0, 0.0

    def draw_visual_map(self, img, all_paths):
        h, w = img.shape[:2]
        # Dibujar zonas
        cv2.polylines(img, [self.config.ORIGIN_ZONE], True, (255, 0, 0), 2, cv2.LINE_AA) 
        cv2.polylines(img, [self.config.EXPECTED_PROJECTION_ZONE], True, (0, 255, 255), 2, cv2.LINE_AA) 
        
        counts = {"Flyrock": 0, "Proyeccion": 0, "Fuera": 0}

        for p in all_paths:
            pts = p['path']
            start_point = tuple(map(int, pts[0]))
            end_point = tuple(map(int, pts[-1]))
            ex, ey = end_point

            # Clasificación
            if ex <= self.config.EDGE_MARGIN or ex >= (w - self.config.EDGE_MARGIN) or \
               ey <= self.config.EDGE_MARGIN or ey >= (h - self.config.EDGE_MARGIN):
                color = (255, 0, 255)
                counts["Fuera"] += 1
            else:
                is_inside = cv2.pointPolygonTest(self.config.EXPECTED_PROJECTION_ZONE, (float(ex), float(ey)), False) >= 0
                color = (0, 165, 255) if is_inside else (0, 0, 255)
                counts["Proyeccion" if is_inside else "Flyrock"] += 1

            # Dibujo de trayectoria parabólica (visual)
            dist_px = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            ctrl_x, ctrl_y = (start_point[0] + end_point[0]) / 2, min(start_point[1], end_point[1]) - (dist_px * 0.3)
            
            t = np.linspace(0, 1, 30)
            curve_pts = []
            for i in t:
                px = (1-i)**2 * start_point[0] + 2*(1-i)*i*ctrl_x + i**2 * end_point[0]
                py = (1-i)**2 * start_point[1] + 2*(1-i)*i*ctrl_y + i**2 * end_point[1]
                curve_pts.append([px, py])
            
            cv2.polylines(img, [np.array(curve_pts, np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.circle(img, start_point, 4, (0, 255, 0), -1)
            cv2.drawMarker(img, end_point, color, cv2.MARKER_CROSS, 12, 2)

        # Labels
        y_offset = 60
        for label, count in [("Seguro", counts["Proyeccion"]), ("PELIGRO", counts["Flyrock"]), ("Incognita", counts["Fuera"])]:
            cv2.putText(img, f"{label}: {count}", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            y_offset += 50

        return img

    def export_to_json(self, all_paths, img_shape, filename="trayectorias_eventos.json"):
        h, w = img_shape[:2]
        json_data = []

        for p in all_paths:
            pts = p['path']
            x1, y1 = pts[0]
            x2, y2 = pts[-1]

            # 1. Distancia en Píxeles
            dist_pixeles = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # 2. DISTANCIA REAL EN METROS (CORREGIDO)
            # Transformamos ambos puntos al espacio de la mina antes de calcular la distancia
            m1 = self.apply_homography((x1, y1))
            m2 = self.apply_homography((x2, y2))
            
            dist_metros = np.sqrt((m2[0] - m1[0])**2 + (m2[1] - m1[1])**2)

            # 3. Clasificación
            is_outside = x2 <= self.config.EDGE_MARGIN or x2 >= (w - self.config.EDGE_MARGIN) or \
                         y2 <= self.config.EDGE_MARGIN or y2 >= (h - self.config.EDGE_MARGIN)
            
            if is_outside:
                clasificacion = "Fuera de Vista"
            else:
                is_inside = cv2.pointPolygonTest(self.config.EXPECTED_PROJECTION_ZONE, (float(x2), float(y2)), False) >= 0
                clasificacion = "Proyeccion" if is_inside else "Flyrock"

            # 4. Generar 1000 puntos para la trayectoria
            ctrl_x, ctrl_y = (x1 + x2) / 2, min(y1, y2) - (dist_pixeles * 0.3)
            t_1000 = np.linspace(0, 1, 1000)
            
            puntos_flat = []
            for i in t_1000:
                px = (1-i)**2 * x1 + 2*(1-i)*i*ctrl_x + i**2 * x2
                py = (1-i)**2 * y1 + 2*(1-i)*i*ctrl_y + i**2 * y2
                puntos_flat.extend([round(float(px), 2), round(float(py), 2)])

            json_data.append({
                "id_roca": p['id'],
                "clasificacion": clasificacion,
                "punto_origen_px": [int(x1), int(y1)],
                "punto_final_px": [int(x2), int(y2)],
                "coords_mina_inicio": [round(m1[0], 2), round(m1[1], 2)],
                "coords_mina_fin": [round(m2[0], 2), round(m2[1], 2)],
                "distancia_pixeles": round(float(dist_pixeles), 2),
                "distancia_metros": round(float(dist_metros), 2),
                "puntos_trayectoria": puntos_flat
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        print(f"[+] Exportación completa: {filename} ({len(json_data)} rocas)")