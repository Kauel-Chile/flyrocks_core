import cv2
import numpy as np
from collections import deque

class RockTracker:
    __slots__ = ('id', 'history', 'frames_since_seen', 'total_detections', 
                 'is_confirmed', 'best_valid_path', 'min_dist', 'kalman', 'config') 

    def __init__(self, id, initial_point, config):
        self.config = config
        self.id = id
        self.history = deque(maxlen=None)
        self.frames_since_seen = 0
        self.total_detections = 1 
        self.is_confirmed = False
        self.best_valid_path = []
        self.min_dist = config.MIN_DISPLACEMENT_BASE 
        
        self.kalman = cv2.KalmanFilter(6, 2)
        self.kalman.measurementMatrix = np.eye(2, 6, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0], [0, 1, 0, 1, 0, 0.5], 
            [0, 0, 1, 0, 1, 0],   [0, 0, 0, 1, 0, 1],   
            [0, 0, 0, 0, 1, 0],   [0, 0, 0, 0, 0, 1]    
        ], np.float32)
        
        self.kalman.statePost = np.array([
            [initial_point[0]], [initial_point[1]], 
            [0], [0], [0], [config.GRAVEDAD_PIXELS]
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
        
        if self.total_detections >= self.config.MIN_FRAMES_TO_CONFIRM:
            self.is_confirmed = True

        if self.is_confirmed and self.is_physically_valid():
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
        
        # 1. Tu lógica original (Magnitud y Tortuosidad)
        displacement = np.linalg.norm(path[-1] - path[0])
        if displacement < self.min_dist: return False
        
        steps = np.diff(path, axis=0)
        total_len = np.sum(np.linalg.norm(steps, axis=1))
        if (total_len / displacement) > self.config.MAX_TORTUOSITY: return False

        # 2. VALIDACIÓN DE CAÍDA VERTICAL (El filtro de talud)
        dy_total = path[-1][1] - path[0][1] # Positivo si terminó más abajo
        dx_total = abs(path[-1][0] - path[0][0]) # Valor absoluto del mov. lateral

        # Solo evaluamos esto si el objeto terminó más abajo de donde empezó
        if dy_total > 0:
            # Evitar división por cero
            if dx_total == 0: 
                return False 
                
            fall_ratio = dy_total / dx_total
            if fall_ratio > self.config.MAX_FALL_RATIO:
                return False

        return True