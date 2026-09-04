import cv2
import numpy as np
import logging
from typing import Any, Dict
from .base import PipelineNode

logger = logging.getLogger(__name__)

class TrajectoryCategorizationNode(PipelineNode):
    
    def __init__(self, name: str = "11_TrajectoryCategorizer", margin_px: int = 5, extrapolation_frames: int = 3):
        super().__init__(name)
        self.margin_px = margin_px
        # Cuántos frames de inercia le perdonamos a una roca para asumir que salió del video
        self.extrapolation_frames = extrapolation_frames 

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Iniciando categorización con extrapolación cinemática...")
                
        trajectories = context.get("filtered_rocks_dict", {}) 
        safety_zone_raw = context.get("expected_projection_zone")
        video_path = context.get("video_path")
        h_matrix_raw = context.get("h_matrix")

        if not trajectories or not safety_zone_raw or not video_path or not h_matrix_raw:
            context["error"] = "Faltan datos en el contexto para la categorización."
            return context

        safety_polygon = np.array(safety_zone_raw, dtype=np.int32).reshape((-1, 1, 2))
        h_matrix = np.array(h_matrix_raw, dtype=np.float64).reshape(3, 3)
        
        try:
            h_inv = np.linalg.inv(h_matrix)
        except np.linalg.LinAlgError:
            context["error"] = "La matriz de homografía proporcionada no es invertible."
            return context

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        resultados_json = {}
        min_x, max_x = self.margin_px, width - self.margin_px
        min_y, max_y = self.margin_px, height - self.margin_px

        for track_id, traj_array in trajectories.items():
            puntos_lista = traj_array[:, 1:3].astype(int).flatten().tolist()
            if not puntos_lista: continue

            frames_lista = traj_array[:, 3].astype(int).tolist()
            n_puntos = len(traj_array)

            first_x, first_y = float(traj_array[0, 1]), float(traj_array[0, 2])
            last_x, last_y = float(traj_array[-1, 1]), float(traj_array[-1, 2])
            
            pts_px = np.array([[first_x, first_y], [last_x, last_y]], dtype=np.float32).reshape(-1, 1, 2)
            pts_metros = cv2.perspectiveTransform(pts_px, h_inv)
            
            real_first_x, real_first_y = pts_metros[0][0]
            real_last_x, real_last_y = pts_metros[1][0]
            distancia_m = float(np.hypot(real_last_x - real_first_x, real_last_y - real_first_y))

            is_out_of_view = False

            # 1. Chequeo estático tradicional (La roca frenó literalmente tocando el borde)
            if last_x <= min_x or last_x >= max_x or last_y <= min_y or last_y >= max_y:
                is_out_of_view = True
            
            # 2. Chequeo predictivo por velocidad (Compuerta de escape)
            elif n_puntos >= 2:
                # Tomamos un promedio de la velocidad en el último tramo (hasta 3 puntos atrás) para evitar ruido
                pts_eval = min(3, n_puntos) 
                prev_x, prev_y = float(traj_array[-pts_eval, 1]), float(traj_array[-pts_eval, 2])
                frames_diff = float(traj_array[-1, 3] - traj_array[-pts_eval, 3])
                
                if frames_diff > 0:
                    vx = (last_x - prev_x) / frames_diff
                    vy = (last_y - prev_y) / frames_diff
                    
                    # Proyectamos la posición basándonos en la velocidad terminal
                    fut_x = last_x + (vx * self.extrapolation_frames)
                    fut_y = last_y + (vy * self.extrapolation_frames)
                    
                    # Si esa inercia lo sacaba del video, entonces es "Fuera de vista"
                    if fut_x <= 0 or fut_x >= width or fut_y <= 0 or fut_y >= height:
                        is_out_of_view = True

            # Asignación final de la categoría
            if is_out_of_view:
                categoria = "Fuera de vista"
            else:
                distancia_poly = cv2.pointPolygonTest(safety_polygon, (last_x, last_y), measureDist=False)
                categoria = "Proyección" if distancia_poly >= 0 else "Proyección peligrosa"

            resultados_json[str(track_id)] = {
                "clasificacion": categoria,
                "distancia_m": round(distancia_m, 2),
                "puntos": puntos_lista,
                "frames": frames_lista
            }

        context["json_resultados"] = resultados_json
        return context