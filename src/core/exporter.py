import cv2
import numpy as np
import json

class ResultsExporter:
    def __init__(self, config):
        self.config = config

    def apply_homography(self, pt, H):
        p = np.array([pt[0], pt[1], 1.0])
        P = np.dot(H, p)
        return P[0] / P[2], P[1] / P[2]

    def draw_visual_map(self, img, all_paths):
        h, w = img.shape[:2]
        cv2.polylines(img, [self.config.ORIGIN_ZONE], True, (255, 0, 0), 2, cv2.LINE_AA) 
        cv2.polylines(img, [self.config.EXPECTED_PROJECTION_ZONE], True, (0, 255, 255), 2, cv2.LINE_AA) 
        
        flyrocks_count = 0
        projections_count = 0
        out_of_bounds_count = 0

        for p in all_paths:
            pts = p['path']
            start_point = tuple(pts[0])   
            end_point = tuple(pts[-1])    
            ex, ey = end_point

            if ex <=  self.config.EDGE_MARGIN or ex >= (w - self.config.EDGE_MARGIN) or ey <= self.config.EDGE_MARGIN or ey >= (h - self.config.EDGE_MARGIN):
                curve_color = (255, 0, 255)  
                marker_color = (255, 0, 255) 
                out_of_bounds_count += 1
            else:
                is_inside = cv2.pointPolygonTest(self.config.EXPECTED_PROJECTION_ZONE, (float(ex), float(ey)), False) >= 0
                if is_inside:
                    curve_color = (0, 165, 255)  
                    marker_color = (0, 165, 255) 
                    projections_count += 1
                else:
                    curve_color = (0, 0, 255)  
                    marker_color = (0, 0, 255) 
                    flyrocks_count += 1

            x1, y1 = start_point
            x2, y2 = end_point
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            ctrl_x, ctrl_y = (x1 + x2) / 2, min(y1, y2) - (dist * 0.3)
            t = np.linspace(0, 1, 50) 
            curve_x = (1 - t)**2 * x1 + 2 * (1 - t) * t * ctrl_x + t**2 * x2
            curve_y = (1 - t)**2 * y1 + 2 * (1 - t) * t * ctrl_y + t**2 * y2
            curve_pts = np.dstack((curve_x, curve_y)).astype(np.int32)

            cv2.polylines(img, curve_pts, False, curve_color, 2, cv2.LINE_AA)
            cv2.circle(img, start_point, 4, (0, 255, 0), -1) 
            cv2.drawMarker(img, end_point, marker_color, cv2.MARKER_CROSS, 12, 2)    

        cv2.putText(img, f"Proyecciones (Seguro): {projections_count}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"FLYROCKS (Peligro): {flyrocks_count}", (40, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"Fuera de Vista (Incognita): {out_of_bounds_count}", (40, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3, cv2.LINE_AA)

        return img

    def export_to_json(self, all_paths, img_shape, filename="trayectorias_eventos.json"):
        h, w = img_shape[:2]
        json_data = []

        for p in all_paths:
            pts = p['path']
            x1, y1 = tuple(pts[0])   
            x2, y2 = tuple(pts[-1])    

            # 1. Distancia Euclidiana en Píxeles
            dist_pixeles = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # 2. Distancia Euclidiana en Metros (usando la Homografía)
            X1_m, Y1_m = self.apply_homography((x1, y1), self.config.H_MATRIX)
            X2_m, Y2_m = self.apply_homography((x2, y2), self.config.H_MATRIX)
            dist_metros = np.sqrt((X2_m - X1_m)**2 + (Y2_m - Y1_m)**2)

            # 3. Clasificación
            clasificacion = ""
            if x2 <= self.config.EDGE_MARGIN or x2 >= (w - self.config.EDGE_MARGIN) or y2 <= self.config.EDGE_MARGIN or y2 >= (h - self.config.EDGE_MARGIN):
                clasificacion = "Fuera de Vista"
            else:
                is_inside = cv2.pointPolygonTest(self.config.EXPECTED_PROJECTION_ZONE, (float(x2), float(y2)), False) >= 0
                clasificacion = "Proyeccion" if is_inside else "Flyrock"

            # 4. Calcular exactamente 1000 puntos para la trayectoria en la imagen
            ctrl_x, ctrl_y = (x1 + x2) / 2, min(y1, y2) - (dist_pixeles * 0.3)
            t_1000 = np.linspace(0, 1, 1000)
            
            curve_x = (1 - t_1000)**2 * x1 + 2 * (1 - t_1000) * t_1000 * ctrl_x + t_1000**2 * x2
            curve_y = (1 - t_1000)**2 * y1 + 2 * (1 - t_1000) * t_1000 * ctrl_y + t_1000**2 * y2
            
            puntos_1000 = []
            for px, py in zip(curve_x, curve_y):
                puntos_1000.append(round(float(px), 2))
                puntos_1000.append(round(float(py), 2))
            

            # 5. Crear el objeto
            trayectoria_obj = {
                "id_roca": p['id'],
                "clasificacion": clasificacion,
                "punto_origen": [int(x1), int(y1)],
                "punto_final": [int(x2), int(y2)],
                "distancia_pixeles": round(float(dist_pixeles), 2),
                "distancia_metros": round(float(dist_metros), 2),  # NUEVO CAMPO
                "puntos_trayectoria": puntos_1000
            }
            
            json_data.append(trayectoria_obj)

        # 6. Guardar el archivo JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        print(f"\n[+] Datos exportados exitosamente a '{filename}'")