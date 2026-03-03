import cv2
import numpy as np
import json  
from scipy.optimize import linear_sum_assignment
from collections import deque
from tqdm import tqdm  

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
VIDEO_PATH = 'data/1/corte_video.mp4'

# Ajustes de rendimiento
STAB_SCALE = 0.4  

# --- PARÁMETROS BASE ---
MIN_FRAMES_TO_CONFIRM = 5  
MIN_AREA = 15        
MAX_AREA = 2500    
DIST_THRESH = 350  
MAX_MISSING = 40   
MIN_SOLIDITY = 0.70  

# --- PARÁMETROS DE FÍSICA ---
MIN_DISPLACEMENT_BASE = 50  
MAX_TORTUOSITY = 1.3  
GRAVEDAD_PIXELS = 1.2 

ORIGIN_ZONE = np.array([
    [1939,1519], [2469,844], [2379,726], [2227,627], [2190,606],
    [2155,587], [2093,553], [1864,810], [1810,873], [1648,1062],
    [1622,1110], [1526,1298], [1501,1348], [1565,1376], [1762,1451]
], np.int32).reshape((-1, 1, 2))

EXPECTED_PROJECTION_ZONE = np.array([
    [3948, 901], [3107, 187], [2750, 23], [2680, -5], [2644, -19],
    [1781, -356], [918, 324], [852, 377], [580, 598], [462, 734],
    [325, 897], [-652, 2064], [665, 2860], [942, 3005], [2663, 3898]
], np.int32).reshape((-1, 1, 2))

# Matriz de Homografía (Convertida a matriz 3x3 de NumPy)
H_LIST = [ -0.015931589199440533, 0.08273002461280851, -6764.196562873699, 
           0.048673511102109844, -0.0033261794074875787, -4584.529246474536, 
          -0.000013337377566862186, 0.000003317779833310359, 1 ]
H_MATRIX = np.array(H_LIST).reshape(3, 3)

ORB_DETECTOR = cv2.ORB_create(nfeatures=800) 
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ==========================================
# 2. CLASE TRACKER
# ==========================================
class RockTracker:
    __slots__ = ('id', 'history', 'frames_since_seen', 'total_detections', 
                 'is_confirmed', 'best_valid_path', 'min_dist', 'kalman') 

    def __init__(self, id, initial_point, required_dist):
        self.id = id
        self.history = deque(maxlen=None)
        self.frames_since_seen = 0
        self.total_detections = 1 
        self.is_confirmed = False
        self.best_valid_path = []
        self.min_dist = required_dist 
        
        self.kalman = cv2.KalmanFilter(6, 2)
        self.kalman.measurementMatrix = np.eye(2, 6, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5], 
            [0, 0, 1, 0, 1, 0],   [0, 0, 0, 1, 0, 1],   
            [0, 0, 0, 0, 1, 0],   [0, 0, 0, 0, 0, 1]    
        ], np.float32)
        
        self.kalman.statePost = np.array([
            [initial_point[0]], [initial_point[1]], 
            [0], [0], [0], [GRAVEDAD_PIXELS]
        ], np.float32)
        
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.history.append(initial_point)

    def update(self, measurement):
        self.kalman.predict()
        self.kalman.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))
        pt = (int(self.kalman.statePost[0][0]), int(self.kalman.statePost[1][0]))
        self.history.append(pt)
        self.frames_since_seen = 0
        self.total_detections += 1
        
        if self.total_detections >= MIN_FRAMES_TO_CONFIRM:
            self.is_confirmed = True

        if self.is_confirmed:
            if self.is_physically_valid(): 
                self.best_valid_path = list(self.history)
        return pt

    def predict_only(self):
        pred = self.kalman.predict()
        pt = (int(pred[0][0]), int(pred[1][0]))
        self.history.append(pt)
        self.frames_since_seen += 1
        return pt

    def is_physically_valid(self):
        if len(self.history) < 8: return False
        path = np.array(self.history)
        displacement = np.linalg.norm(path[-1] - path[0])
        if displacement < self.min_dist: return False
        
        steps = np.diff(path, axis=0)
        total_len = np.sum(np.linalg.norm(steps, axis=1))
        return (total_len / displacement) <= MAX_TORTUOSITY

# ==========================================
# 3. ESTABILIZACIÓN OPTIMIZADA
# ==========================================
def get_fast_stabilization_matrix(ref_gray, curr_gray):
    h, w = ref_gray.shape
    new_dim = (int(w * STAB_SCALE), int(h * STAB_SCALE))
    
    ref_small = cv2.resize(ref_gray, new_dim, interpolation=cv2.INTER_LINEAR)
    curr_small = cv2.resize(curr_gray, new_dim, interpolation=cv2.INTER_LINEAR)

    kp1, des1 = ORB_DETECTOR.detectAndCompute(ref_small, None)
    kp2, des2 = ORB_DETECTOR.detectAndCompute(curr_small, None)
    
    if des1 is None or des2 is None: return None
    matches = MATCHER.match(des1, des2)
    if len(matches) < 15: return None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    matrix_small, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
    if matrix_small is None: return None

    matrix_full = matrix_small.copy()
    matrix_full[0, 2] /= STAB_SCALE
    matrix_full[1, 2] /= STAB_SCALE
    
    return matrix_full

# ==========================================
# 4. MAPA VISUAL Y EXPORTACIÓN JSON
# ==========================================
def draw_visual_map(img, all_paths):
    h, w = img.shape[:2]
    EDGE_MARGIN = 5

    cv2.polylines(img, [ORIGIN_ZONE], True, (255, 0, 0), 2, cv2.LINE_AA) 
    cv2.polylines(img, [EXPECTED_PROJECTION_ZONE], True, (0, 255, 255), 2, cv2.LINE_AA) 
    
    flyrocks_count = 0
    projections_count = 0
    out_of_bounds_count = 0

    for p in all_paths:
        pts = p['path']
        start_point = tuple(pts[0])   
        end_point = tuple(pts[-1])    
        ex, ey = end_point

        if ex <= EDGE_MARGIN or ex >= (w - EDGE_MARGIN) or ey <= EDGE_MARGIN or ey >= (h - EDGE_MARGIN):
            curve_color = (255, 0, 255)  
            marker_color = (255, 0, 255) 
            out_of_bounds_count += 1
        else:
            is_inside = cv2.pointPolygonTest(EXPECTED_PROJECTION_ZONE, (float(ex), float(ey)), False) >= 0
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

def apply_homography(pt, H):
    """Aplica la matriz de homografía a un punto 2D (x, y) para obtener coordenadas del mundo real."""
    p = np.array([pt[0], pt[1], 1.0])
    P = np.dot(H, p)
    return P[0] / P[2], P[1] / P[2]

def export_to_json(all_paths, img_shape, filename="trayectorias_eventos.json"):
    h, w = img_shape[:2]
    EDGE_MARGIN = 5
    json_data = []

    for p in all_paths:
        pts = p['path']
        x1, y1 = tuple(pts[0])   
        x2, y2 = tuple(pts[-1])    

        # 1. Distancia Euclidiana en Píxeles
        dist_pixeles = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # 2. Distancia Euclidiana en Metros (usando la Homografía)
        X1_m, Y1_m = apply_homography((x1, y1), H_MATRIX)
        X2_m, Y2_m = apply_homography((x2, y2), H_MATRIX)
        dist_metros = np.sqrt((X2_m - X1_m)**2 + (Y2_m - Y1_m)**2)

        # 3. Clasificación
        clasificacion = ""
        if x2 <= EDGE_MARGIN or x2 >= (w - EDGE_MARGIN) or y2 <= EDGE_MARGIN or y2 >= (h - EDGE_MARGIN):
            clasificacion = "Fuera de Vista"
        else:
            is_inside = cv2.pointPolygonTest(EXPECTED_PROJECTION_ZONE, (float(x2), float(y2)), False) >= 0
            clasificacion = "Proyeccion" if is_inside else "Flyrock"

        # 4. Calcular exactamente 1000 puntos para la trayectoria en la imagen
        ctrl_x, ctrl_y = (x1 + x2) / 2, min(y1, y2) - (dist_pixeles * 0.3)
        t_1000 = np.linspace(0, 1, 1000)
        
        curve_x = (1 - t_1000)**2 * x1 + 2 * (1 - t_1000) * t_1000 * ctrl_x + t_1000**2 * x2
        curve_y = (1 - t_1000)**2 * y1 + 2 * (1 - t_1000) * t_1000 * ctrl_y + t_1000**2 * y2
        
        # Formatear la lista de 1000 puntos (x, y) redondeados a 2 decimales
        puntos_1000 = [[round(float(px), 2), round(float(py), 2)] for px, py in zip(curve_x, curve_y)]

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

# ==========================================
# 5. MAIN
# ==========================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret: return
    
    map_initial = first_frame.copy()
    last_valid_frame = first_frame.copy()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    backSub = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=80, detectShadows=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    kernel_3 = np.ones((3,3), np.uint8)

    trackers = []
    all_paths = []
    track_id_count = 0
    
    print(f"Procesando {total_frames} frames...")
    pbar = tqdm(total=total_frames, unit="frames", desc="Tracking Flyrocks")
    pbar.update(1) 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        last_valid_frame = frame 
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        M = get_fast_stabilization_matrix(prev_gray, curr_gray)
        if M is not None:
            frame_stab = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            gray_stab = cv2.warpAffine(curr_gray, M, (frame.shape[1], frame.shape[0]))
        else:
            frame_stab, gray_stab = frame, curr_gray

        gray_filtered = cv2.medianBlur(gray_stab, 5)
        gray_enhanced = clahe.apply(gray_filtered)
        fgMask = backSub.apply(gray_enhanced, learningRate=0.01)

        _, fgMask = cv2.threshold(fgMask, 240, 255, cv2.THRESH_BINARY)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_3, iterations=1)
        fgMask = cv2.dilate(fgMask, kernel_3, iterations=1)

        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and (area / hull_area) > MIN_SOLIDITY:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append((int(x + w/2), int(y + h/2)))

        if trackers and detections:
            T_pos = np.array([(t.kalman.statePost[0][0], t.kalman.statePost[1][0]) for t in trackers])
            D_pos = np.array(detections)
            cost = np.linalg.norm(T_pos[:, np.newaxis, :] - D_pos[np.newaxis, :, :], axis=2)
            row, col = linear_sum_assignment(cost)
            assigned_trackers, assigned_detections = set(), set()
            for r, c in zip(row, col):
                if cost[r, c] < DIST_THRESH:
                    trackers[r].update(detections[c])
                    assigned_trackers.add(r)
                    assigned_detections.add(c)
        else:
            assigned_trackers, assigned_detections = set(), set()

        for i, det in enumerate(detections):
            if i not in assigned_detections:
                if cv2.pointPolygonTest(ORIGIN_ZONE, (float(det[0]), float(det[1])), False) >= 0:
                    trackers.append(RockTracker(track_id_count, det, MIN_DISPLACEMENT_BASE))
                    track_id_count += 1

        alive = []
        for i, t in enumerate(trackers):
            if i not in assigned_trackers: t.predict_only()
            
            pos = t.kalman.statePost
            if (t.frames_since_seen <= MAX_MISSING and 
                0 < pos[0][0] < frame.shape[1] and 
                0 < pos[1][0] < frame.shape[0]):
                alive.append(t)
            elif t.is_confirmed and t.best_valid_path:
                all_paths.append({'path': t.best_valid_path, 'id': t.id})
        trackers = alive
        
        prev_gray = gray_stab.copy()
        pbar.update(1) 

    pbar.close()
    
    # ==========================================
    # 5. GENERACIÓN RESULTADOS
    # ==========================================
    final_frame_clean = last_valid_frame.copy()

    for t in trackers:
        if t.is_confirmed and t.best_valid_path:
            all_paths.append({'path': t.best_valid_path, 'id': t.id})

    print(f"\nGenerando mapas visuales con {len(all_paths)} eventos registrados...")
    cv2.imwrite("mapa_impactos_inicio.jpg", draw_visual_map(map_initial, all_paths))
    cv2.imwrite("mapa_impactos_final.jpg", draw_visual_map(final_frame_clean, all_paths))
    
    # Exportar el nuevo archivo JSON
    export_to_json(all_paths, final_frame_clean.shape, "trayectorias_eventos.json")

    print("¡Proceso completado! Revisa las imágenes y el archivo JSON generados.")
    cap.release()

if __name__ == "__main__":
    main()