import numpy as np
import cv2
from PIL import Image

def compare_deformation(img_q: Image.Image, img_r: Image.Image) -> dict:
    """제품의 Solidity(치밀도) 차이를 분석하여 찌그러짐을 감지합니다."""
    def get_shape_features(img):
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return 0.0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        return solidity

    sol_q = get_shape_features(img_q)
    sol_r = get_shape_features(img_r)
    
    return {
        "solidity_L": sol_q, # 불량(Query)
        "solidity_R": sol_r, # 정상(Match)
        "delta_dent": sol_r - sol_q, # 양수일수록 불량이 더 찌그러짐
        "delta_edge": abs(sol_q - sol_r)
    }