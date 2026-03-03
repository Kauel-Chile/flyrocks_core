import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.core.config import Config
from src.core.vision import VisionProcessor
from src.core.tracker import RockTracker
from src.core.exporter import ResultsExporter

def main():
    config = Config()
    vision = VisionProcessor(config)
    exporter = ResultsExporter(config)

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret: return
    
    map_initial = first_frame.copy()
    last_valid_frame = first_frame.copy()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    trackers = []
    all_paths = []
    track_id_count = 0
    
    print(f"Procesando {total_frames} frames...")
    pbar = tqdm(total=total_frames, unit="frames", desc="Tracking Flyrocks")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        last_valid_frame = frame 
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Estabilización
        M = vision.get_fast_stabilization_matrix(prev_gray, curr_gray)
        if M is not None:
            frame_stab = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            gray_stab = cv2.warpAffine(curr_gray, M, (frame.shape[1], frame.shape[0]))
        else:
            frame_stab, gray_stab = frame, curr_gray

        # 2. Detecciones
        detections = vision.extract_detections(gray_stab)

        # 3. Asignación (Húngaro)
        assigned_trackers, assigned_detections = set(), set()
        if trackers and detections:
            T_pos = np.array([(t.kalman.statePost[0][0], t.kalman.statePost[1][0]) for t in trackers])
            D_pos = np.array(detections)
            cost = np.linalg.norm(T_pos[:, np.newaxis, :] - D_pos[np.newaxis, :, :], axis=2)
            row, col = linear_sum_assignment(cost)
            
            for r, c in zip(row, col):
                if cost[r, c] < config.DIST_THRESH:
                    trackers[r].update(detections[c])
                    assigned_trackers.add(r)
                    assigned_detections.add(c)

        # 4. Nuevos trackers
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                if cv2.pointPolygonTest(config.ORIGIN_ZONE, (float(det[0]), float(det[1])), False) >= 0:
                    trackers.append(RockTracker(track_id_count, det, config))
                    track_id_count += 1

        # 5. Mantenimiento de Trackers vivos
        alive = []
        for i, t in enumerate(trackers):
            if i not in assigned_trackers: t.predict_only()
            
            pos = t.kalman.statePost
            if (t.frames_since_seen <= config.MAX_MISSING and 
                0 < pos[0][0] < frame.shape[1] and 
                0 < pos[1][0] < frame.shape[0]):
                alive.append(t)
            elif t.is_confirmed and t.best_valid_path:
                all_paths.append({'path': t.best_valid_path, 'id': t.id})
        trackers = alive
        
        prev_gray = gray_stab.copy()
        pbar.update(1) 

    pbar.close()
    
    # 6. Guardar los que quedaron vivos al final
    for t in trackers:
        if t.is_confirmed and t.best_valid_path:
            all_paths.append({'path': t.best_valid_path, 'id': t.id})

    # 7. Exportación
    print(f"\nGenerando mapas visuales con {len(all_paths)} eventos registrados...")
    cv2.imwrite("mapa_impactos_inicio.jpg", exporter.draw_visual_map(map_initial, all_paths))
    cv2.imwrite("mapa_impactos_final.jpg", exporter.draw_visual_map(last_valid_frame, all_paths))
    exporter.export_to_json(all_paths, last_valid_frame.shape, "trayectorias_eventos.json")

    cap.release()

if __name__ == "__main__":
    main()