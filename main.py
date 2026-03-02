import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from tqdm import tqdm  # Barra de progreso

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
VIDEO_PATH = 'data/7/corte_video.mp4'

# Ajustes de rendimiento
STAB_SCALE = 0.4  # Reducimos la imagen al 40% solo para calcular el movimiento (Mucho más rápido)

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

# Zona de Origen
ORIGIN_ZONE = np.array([
    [1808, 1403], [1981, 1358], [1820, 1028], [1785, 1032], 
    [1712, 1039], [1693, 1041], [1684, 1059], [1688, 1085], 
    [1700, 1141], [1711, 1184]
], np.int32).reshape((-1, 1, 2))


# Detector ORB Global (reutilizable)
ORB_DETECTOR = cv2.ORB_create(nfeatures=800) # 800 es suficiente para estabilización rápida
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ==========================================
# 2. CLASE TRACKER
# ==========================================
class RockTracker:
    __slots__ = ('id', 'history', 'frames_since_seen', 'total_detections', 
                 'is_confirmed', 'best_valid_path', 'min_dist', 'kalman') # Optimización de memoria

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
        # Matriz de transición optimizada
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

        # Validación perezosa: solo calculamos si es necesario
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
        # Cálculo vectorizado rápido usando numpy
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
    """Calcula la matriz en baja resolución y la escala al tamaño original."""
    # 1. Reducir tamaño para velocidad extrema (ORB es lento en HD/4K)
    h, w = ref_gray.shape
    new_dim = (int(w * STAB_SCALE), int(h * STAB_SCALE))
    
    ref_small = cv2.resize(ref_gray, new_dim, interpolation=cv2.INTER_LINEAR)
    curr_small = cv2.resize(curr_gray, new_dim, interpolation=cv2.INTER_LINEAR)

    # 2. Detectar en pequeño
    kp1, des1 = ORB_DETECTOR.detectAndCompute(ref_small, None)
    kp2, des2 = ORB_DETECTOR.detectAndCompute(curr_small, None)
    
    if des1 is None or des2 is None: return None
    matches = MATCHER.match(des1, des2)
    if len(matches) < 15: return None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 3. Calcular matriz parcial
    matrix_small, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
    
    if matrix_small is None: return None

    # 4. Escalar la traslación de vuelta al tamaño original
    # La rotación y escala (indices 0,0; 0,1; 1,0; 1,1) no cambian.
    # Solo las traslaciones (columna 2) necesitan multiplicarse por 1/scale
    matrix_full = matrix_small.copy()
    matrix_full[0, 2] /= STAB_SCALE
    matrix_full[1, 2] /= STAB_SCALE
    
    return matrix_full

def draw_visual_map(img, all_paths):
    for p in all_paths:
        pts = p['path']
        start_point = tuple(pts[0])   
        end_point = tuple(pts[-1])    
        rock_id = p['id']

        # Parábola
        x1, y1 = start_point
        x2, y2 = end_point
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        ctrl_x, ctrl_y = (x1 + x2) / 2, min(y1, y2) - (dist * 0.3)
        t = np.linspace(0, 1, 50)
        curve_x = (1 - t)**2 * x1 + 2 * (1 - t) * t * ctrl_x + t**2 * x2
        curve_y = (1 - t)**2 * y1 + 2 * (1 - t) * t * ctrl_y + t**2 * y2
        curve_pts = np.dstack((curve_x, curve_y)).astype(np.int32)

        cv2.polylines(img, curve_pts, False, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(img, start_point, 5, (0, 255, 0), -1) 
        cv2.drawMarker(img, end_point, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(img, f"#{rock_id}", (end_point[0] + 5, end_point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"#{rock_id}", (end_point[0] + 5, end_point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img

# ==========================================
# 4. MAIN
# ==========================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret: return
    
    # Pre-allocating variables
    map_initial = first_frame.copy()
    last_valid_frame = first_frame.copy()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    backSub = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=80, detectShadows=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    kernel_3 = np.ones((3,3), np.uint8)

    trackers = []
    all_paths = []
    track_id_count = 0
    
    # Inicializar TQDM
    print(f"Procesando {total_frames} frames...")
    pbar = tqdm(total=total_frames, unit="frames", desc="Tracking Flyrocks")
    pbar.update(1) # Contamos el primer frame ya leído

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        last_valid_frame = frame # No hacemos copy() aquí para velocidad, solo al final si es necesario
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Estabilización Rápida
        M = get_fast_stabilization_matrix(prev_gray, curr_gray)
        
        if M is not None:
            # Warp affine sigue siendo en resolución completa para no perder detalle de rocas
            frame_stab = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            gray_stab = cv2.warpAffine(curr_gray, M, (frame.shape[1], frame.shape[0]))
        else:
            frame_stab, gray_stab = frame, curr_gray

        # 2. Detección (Filtros + Sustracción)
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
                # Chequeo rápido de solidez
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and (area / hull_area) > MIN_SOLIDITY:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append((int(x + w/2), int(y + h/2)))

        # 3. Tracking (Hungarian)
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

        # 4. Gestión de Trackers
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                if cv2.pointPolygonTest(ORIGIN_ZONE, (float(det[0]), float(det[1])), False) >= 0:
                    trackers.append(RockTracker(track_id_count, det, MIN_DISPLACEMENT_BASE))
                    track_id_count += 1

        alive = []
        for i, t in enumerate(trackers):
            if i not in assigned_trackers: t.predict_only()
            
            # Chequeo de bordes optimizado
            pos = t.kalman.statePost
            if (t.frames_since_seen <= MAX_MISSING and 
                0 < pos[0][0] < frame.shape[1] and 
                0 < pos[1][0] < frame.shape[0]):
                alive.append(t)
            elif t.is_confirmed and t.best_valid_path:
                all_paths.append({'path': t.best_valid_path, 'id': t.id})
        trackers = alive
        
        prev_gray = gray_stab.copy()
        pbar.update(1) # Actualizar tqdm

    pbar.close()
    
    # ==========================================
    # 5. GENERACIÓN RESULTADOS
    # ==========================================
    # Guardar una copia segura del último frame ahora que terminó el loop
    final_frame_clean = last_valid_frame.copy()

    # Recolectar trackers sobrevivientes
    for t in trackers:
        if t.is_confirmed and t.best_valid_path:
            all_paths.append({'path': t.best_valid_path, 'id': t.id})

    print(f"\nGenerando mapas con {len(all_paths)} eventos...")

    cv2.imwrite("mapa_impactos_inicio.jpg", draw_visual_map(map_initial, all_paths))
    cv2.imwrite("mapa_impactos_final.jpg", draw_visual_map(final_frame_clean, all_paths))
    
    print("¡Proceso completado!")
    cap.release()

if __name__ == "__main__":
    main()