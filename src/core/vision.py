import cv2
import numpy as np

class VisionProcessor:
    def __init__(self, config):
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=800) 
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=80, detectShadows=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.kernel_3 = np.ones((3,3), np.uint8)

    def get_fast_stabilization_matrix(self, ref_gray, curr_gray):
        h, w = ref_gray.shape
        new_dim = (int(w * self.config.STAB_SCALE), int(h * self.config.STAB_SCALE))
        
        ref_small = cv2.resize(ref_gray, new_dim, interpolation=cv2.INTER_LINEAR)
        curr_small = cv2.resize(curr_gray, new_dim, interpolation=cv2.INTER_LINEAR)

        kp1, des1 = self.orb.detectAndCompute(ref_small, None)
        kp2, des2 = self.orb.detectAndCompute(curr_small, None)
        
        if des1 is None or des2 is None: return None
        matches = self.matcher.match(des1, des2)
        if len(matches) < 15: return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        matrix_small, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        if matrix_small is None: return None

        matrix_full = matrix_small.copy()
        matrix_full[0, 2] /= self.config.STAB_SCALE
        matrix_full[1, 2] /= self.config.STAB_SCALE
        
        return matrix_full

    def extract_detections(self, gray_stab):
        gray_filtered = cv2.medianBlur(gray_stab, 5)
        gray_enhanced = self.clahe.apply(gray_filtered)
        fgMask = self.backSub.apply(gray_enhanced, learningRate=0.01)

        _, fgMask = cv2.threshold(fgMask, 240, 255, cv2.THRESH_BINARY)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, self.kernel_3, iterations=1)
        fgMask = cv2.dilate(fgMask, self.kernel_3, iterations=1)

        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.MIN_AREA < area < self.config.MAX_AREA:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and (area / hull_area) > self.config.MIN_SOLIDITY:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append((int(x + w/2), int(y + h/2)))
        return detections