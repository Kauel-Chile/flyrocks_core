import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import PipelineNode

logger = logging.getLogger(__name__)

class EventExtractorNode(PipelineNode):

    cacheable = False

    def __init__(
        self, 
        name: str = "EventExtractor4D",
        noise_threshold: int = 8, 
        blur_kernel: Tuple[int, int] = (3, 3),
        fallback_video_path: Optional[str | Path] = None,
        output_mask_filename: str = "mascara_cambios.png"
    ):
        super().__init__(name)
        self.noise_threshold = noise_threshold
        self.blur_kernel = blur_kernel
        self.fallback_video_path = Path(fallback_video_path) if fallback_video_path else None
        self.output_mask_filename = output_mask_filename
        self.tensor_raw: Optional[np.ndarray] = None

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        path_str = context.get("video_path", self.fallback_video_path)
        if not path_str:
            context["error"] = "Missing 'video_path'"
            return context
            
        video_path = Path(path_str)
        if not video_path.exists():
            context["error"] = f"File not found: {video_path}"
            return context

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            context["error"] = "Failed to open video."
            return context

        event_cloud_4d = []
        frame_index = 2
        
        try:
            ret, previous_frame = cap.read()
            if not ret: return context

            height, width = previous_frame.shape[:2]
            max_change_mask = np.zeros((height, width), dtype=np.uint8)

            # CONVERSIÓN A INT16: Fundamental para permitir restas con signo negativo
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.GaussianBlur(previous_gray, self.blur_kernel, 0).astype(np.int16)
            
            while True:
                ret, current_frame = cap.read()
                if not ret: break

                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                current_gray = cv2.GaussianBlur(current_gray, self.blur_kernel, 0).astype(np.int16)
                
                # Diferencia matemática real (con signo)
                difference = current_gray - previous_gray
                
                # Matriz absoluta requerida para la UI visual y el thresholding
                abs_diff = np.abs(difference)
                
                cv2.max(max_change_mask, abs_diff.astype(np.uint8), dst=max_change_mask)

                ys, xs = np.nonzero(abs_diff > self.noise_threshold)
                
                if ys.size > 0:
                    # EXTRAER INTENSIDADES ORIGINALES: Se sacan de "difference", no de "abs_diff"
                    intensities = difference[ys, xs]
                    
                    # DOWNCASTING A INT16: Conserva memoria y signos en una sola matriz
                    xs = xs.astype(np.int16)
                    ys = ys.astype(np.int16)
                    ts = np.full(ys.size, frame_index, dtype=np.int16)
                    
                    frame_events = np.column_stack((xs, ys, ts, intensities))
                    event_cloud_4d.append(frame_events)

                previous_gray = current_gray
                frame_index += 1

            v_max = int(np.max(max_change_mask))
            if 0 < v_max < 255:
                alpha = 255 - v_max
                mask_activa = (max_change_mask > 0).astype(np.uint8)
                cv2.add(max_change_mask, alpha, dst=max_change_mask, mask=mask_activa)
                logger.info(f"[{self.name}] Máscara mejorada con traslación alpha={alpha}")

            mask_path = video_path.parent / self.output_mask_filename
            cv2.imwrite(str(mask_path), max_change_mask)
            context["change_mask_path"] = str(mask_path)
            logger.info(f"[{self.name}] Máscara de cambios absolutos guardada en: {mask_path}")

        except Exception as e:
            context["error"] = str(e)
        finally:
            logger.info(f"[{self.name}] Frames totales procesados: {frame_index}")
            cap.release()

        if event_cloud_4d:
            self.tensor_raw = np.concatenate(event_cloud_4d, axis=0)
            context["tensor_raw"] = self.tensor_raw
            logger.info(f"[{self.name}] Extracted {len(self.tensor_raw):,} SIGNED events (Dtype: {self.tensor_raw.dtype}).")
        else:
            context["tensor_raw"] = None

        return context