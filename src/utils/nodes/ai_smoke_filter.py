import numpy as np
import cv2
import logging
import math
import onnxruntime as ort
from typing import Any, Dict

from .base import PipelineNode

logger = logging.getLogger(__name__)

class AISmokeFilterNode(PipelineNode):
    """
    Proyecta ventanas temporales del tensor de eventos a 2D y utiliza
    una red neuronal ONNX para identificar y eliminar puntos asociados al humo.
    Soporta tensores int16 con signo aplicando magnitud absoluta sobre el canvas.
    """
    def __init__(
        self, 
        name: str = "AISmokeFilter", 
        onnx_path: str = "detovision_model_v18.onnx", 
        frames_contexto: int = 60, 
        avance_frames: int = 30, 
        umbral_prob: float = 0.90,
        escala: float = 0.75
    ):
        super().__init__(name)
        self.onnx_path = onnx_path
        self.frames_contexto = frames_contexto
        self.avance_frames = avance_frames
        self.umbral_prob = umbral_prob
        self.escala = escala
        
        self._session = None
        self._input_name = None

    def _obtener_probabilidad_humo(self, image_gray):
        h_orig, w_orig = image_gray.shape
        
        w_nuevo, h_nuevo = int(w_orig * self.escala), int(h_orig * self.escala)
        factor = math.ceil(1.0 / self.escala) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (factor, factor))
        
        img_dilatada = cv2.dilate(image_gray, kernel)
        img_reducida = cv2.resize(img_dilatada, (w_nuevo, h_nuevo), interpolation=cv2.INTER_NEAREST)
        img_h, img_w = img_reducida.shape
        
        pad_w = (128 - (img_w % 128)) % 128
        pad_h = (128 - (img_h % 128)) % 128
        
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(img_reducida, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        else:
            padded = img_reducida
            
        input_tensor = padded.astype(np.float32)
        input_tensor /= 255.0  
        input_tensor = np.expand_dims(input_tensor, axis=(0, 1))
        
        logits = self._session.run(None, {self._input_name: input_tensor})[0][0]
        logits_cropped = logits[:, :img_h, :img_w]
        
        diff = logits_cropped[0, :, :] - logits_cropped[1, :, :]
        np.exp(diff, out=diff)      
        diff += 1.0                 
        np.reciprocal(diff, out=diff) 
        
        prob_humo_restaurada = cv2.resize(diff, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return prob_humo_restaurada

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tensor = context.get("tensor_raw")
        if tensor is None or len(tensor) == 0:
            logger.warning(f"[{self.name}] No se encontró 'tensor_raw' o está vacío. Omitiendo.")
            return context

        # FUERZA INT16 PARA SOPORTE DE SIGNOS
        if tensor.dtype != np.int16:
            tensor = tensor.astype(np.int16)

        if self._session is None:
            try:
                self._session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
                self._input_name = self._session.get_inputs()[0].name
            except Exception as e:
                context["error"] = f"Error al cargar el modelo ONNX en {self.onnx_path}: {e}"
                return context

        max_x = int(np.max(tensor[:, 0])) + 1
        max_y = int(np.max(tensor[:, 1])) + 1
        max_t = int(np.max(tensor[:, 2])) + 1 
        
        tensor_filtrado = []
        
        # El canvas de inferencia de IA no sabe qué son valores negativos
        canvas = np.zeros((max_y, max_x), dtype=np.uint8)
        
        logger.info(f"[{self.name}] Ejecutando inferencia en ventanas temporales (Escala: {self.escala*100:.0f}%)...")

        for start_frame in range(0, max_t, self.avance_frames):
            end_frame_ctx = start_frame + self.frames_contexto
            
            puntos_ctx = tensor[(tensor[:, 2] >= start_frame) & (tensor[:, 2] < end_frame_ctx)]
            
            canvas.fill(0)
            if len(puntos_ctx) > 0:
                y_c = puntos_ctx[:, 1].astype(int)
                x_c = puntos_ctx[:, 0].astype(int)
                
                # ADAPTACIÓN SEGURA: Extraemos magnitud (abs), aplicamos clip y forzamos a uint8 
                # Esto protege el canvas sin modificar el tensor_raw original
                intensidades_abs = np.clip(np.abs(puntos_ctx[:, 3]), 0, 255).astype(np.uint8)
                np.maximum.at(canvas, (y_c, x_c), intensidades_abs)
            
            prob_humo = self._obtener_probabilidad_humo(canvas)
            
            end_frame_guardado = min(start_frame + self.avance_frames, max_t)
            puntos_app = tensor[(tensor[:, 2] >= start_frame) & (tensor[:, 2] < end_frame_guardado)]
            
            y_idx = puntos_app[:, 1].astype(int)
            x_idx = puntos_app[:, 0].astype(int)

            mask_bounds = (y_idx < prob_humo.shape[0]) & (x_idx < prob_humo.shape[1])
            y_valid = y_idx[mask_bounds]
            x_valid = x_idx[mask_bounds]

            mask_prob = prob_humo[y_valid, x_valid] < self.umbral_prob

            chunk_valido = puntos_app[mask_bounds][mask_prob]
            tensor_filtrado.append(chunk_valido)
            
            del prob_humo, puntos_ctx, puntos_app, x_idx, y_idx, mask_bounds, y_valid, x_valid, mask_prob
            
        tensor_final = np.vstack(tensor_filtrado) if tensor_filtrado else np.empty((0, 4), dtype=np.int16)
        
        retencion = (len(tensor_final) / len(tensor)) * 100 if len(tensor) > 0 else 0
        logger.info(f"[{self.name}] Filtrado completado. Se conservaron {len(tensor_final)} pts ({retencion:.1f}%).")
        
        context["tensor_raw"] = tensor_final
        return context