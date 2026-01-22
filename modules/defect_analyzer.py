import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DefectRegion:
    bbox: Tuple[int, int, int, int]
    clock_position: str
    defect_type: str
    severity: float
    confidence: float
    sequence_score: float
    center_coords: Tuple[int, int]
    area: float

class AdvancedDefectAnalyzer:
    def __init__(self, min_defect_area=25, confidence_threshold=0.15, 
                 intensity_threshold=8, max_defect_area=15000, grid_size=7):
        """
        [Michael님 현장 맞춤형 파라미터]
        - min_defect_area: 25 (매우 작은 찍힘 점까지 포착)
        - intensity_threshold: 8 (육안으로 흐릿한 음영까지 분석 대상 포함)
        """
        self.min_defect_area = min_defect_area
        self.confidence_threshold = confidence_threshold
        self.intensity_threshold = intensity_threshold
        self.max_defect_area = max_defect_area
        self.grid_size = grid_size

    def _enhance_contrast(self, img_cv: np.ndarray) -> np.ndarray:
        """CLAHE 강도를 높여 미세한 흠집의 명암비를 극대화"""
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # clipLimit을 4.0으로 상향하여 대비를 더 강하게 줍니다.
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def extract_precise_mask(self, img: np.ndarray) -> np.ndarray:
        """제품 외곽 배경 노이즈를 제거하기 위한 정밀 마스크 (Erode 강화)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 제품 영역 확정
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # [핵심] 배경 경계면 노이즈 방지를 위해 마스크를 안쪽으로 15픽셀 수축(Erode)
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)
        
        # 중앙 날개 구조물 오검출 방지 (중앙 36% 영역 강제 제외)
        h, w = mask.shape
        cv2.circle(mask, (w//2, h//2), int(min(w, h) * 0.36), 0, -1)
        
        return mask

    def _analyze_pixel_sequence(self, diff_roi: np.ndarray) -> float:
        h, w = diff_roi.shape
        cell_h, cell_w = max(1, h // self.grid_size), max(1, w // self.grid_size)
        sequence_data = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = diff_roi[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                if cell.size > 0:
                    sequence_data.append(np.mean(cell))
        
        continuity_count = 0
        for k in range(1, len(sequence_data)):
            if sequence_data[k] > self.intensity_threshold and sequence_data[k-1] > self.intensity_threshold:
                continuity_count += 1
        return continuity_count / (self.grid_size ** 2)

    def classify_defect_sequential(self, contour, diff_roi):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: return "미세 노이즈", 0.1, 0.1
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        seq_score = self._analyze_pixel_sequence(diff_roi)
        mean_intensity = np.mean(diff_roi) if diff_roi.size > 0 else 0
        
        # 찍힘(Dent) 기준 완화하여 더 잘 잡히도록 수정
        if (circularity > 0.35 and area < 1000) or (mean_intensity > self.intensity_threshold * 1.3):
            return "찍힘(Dent)", min(1.0, mean_intensity / 35.0), seq_score
        elif seq_score > 0.25 or (aspect_ratio > 3.0 or aspect_ratio < 0.33):
            return "스크래치(Scratch)", min(1.0, seq_score * 2.0), seq_score
        return "표면 변화", 0.15, seq_score
        
    def analyze(self, ref_img: Image.Image, tgt_img: Image.Image) -> List[DefectRegion]:
        ref_cv = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2BGR)
        tgt_cv = cv2.cvtColor(np.array(tgt_img), cv2.COLOR_RGB2BGR)
        tgt_cv = cv2.resize(tgt_cv, (ref_cv.shape[1], ref_cv.shape[0]))
        
        # 1. 대비 강화
        ref_cv_en = self._enhance_contrast(ref_cv)
        tgt_cv_en = self._enhance_contrast(tgt_cv)
        
        # 2. 정밀 마스크 추출 (배경 노이즈 차단)
        obj_mask = self.extract_precise_mask(tgt_cv)
        
        gray_ref = cv2.cvtColor(ref_cv_en, cv2.COLOR_BGR2GRAY)
        gray_tgt = cv2.cvtColor(tgt_cv_en, cv2.COLOR_BGR2GRAY)
        
        # 3. 차분 및 마스킹
        diff = cv2.absdiff(cv2.GaussianBlur(gray_ref, (3,3), 0), cv2.GaussianBlur(gray_tgt, (3,3), 0))
        masked_diff = cv2.bitwise_and(diff, diff, mask=obj_mask)
        
        # 4. 적응형 이진화 (감도 극대화: C값을 -5로 조정)
        # 
        thresh = cv2.adaptiveThreshold(masked_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 31, -5)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area or area > self.max_defect_area: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            roi = masked_diff[y:y+h, x:x+w]
            
            dtype, conf, s_score = self.classify_defect_sequential(cnt, roi)
            if conf < self.confidence_threshold: continue
            
            defects.append(DefectRegion(
                bbox=(x, y, w, h), 
                clock_position=self.get_clock_pos(x, y, w, h, tgt_cv.shape),
                defect_type=dtype, severity=conf, confidence=conf,
                sequence_score=s_score,
                center_coords=(x+w//2, y+h//2), area=area
            ))
            
        return sorted(defects, key=lambda d: d.area, reverse=True)[:5]

    def get_clock_pos(self, x, y, w, h, shape):
        cx, cy = shape[1]//2, shape[0]//2
        dx, dy = (x+w//2) - cx, cy - (y+h//2)
        angle = np.degrees(np.arctan2(dx, dy)) % 360
        hour = int((angle + 15) / 30) % 12
        return f"{12 if hour == 0 else hour}시"

    def visualize_results(self, ref_img, tgt_img, defects, save_path=None):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        axes[0].imshow(ref_img); axes[0].set_title("Standard (OK)"); axes[0].axis('off')
        
        # 분석 대상 이미지는 대비 강화된 버전으로 보여주어 Michael님이 확인하기 쉽게 함
        tgt_en = self._enhance_contrast(cv2.cvtColor(np.array(tgt_img), cv2.COLOR_RGB2BGR))
        axes[1].imshow(cv2.cvtColor(tgt_en, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Analysis View (Enhanced)"); axes[1].axis('off')
        
        axes[2].imshow(tgt_img)
        for i, d in enumerate(defects, 1):
            x, y, w, h = d.bbox
            axes[2].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
            axes[2].text(x, y-12, f"#{i} {d.clock_position} {d.defect_type}", 
                         color='white', fontsize=9, fontweight='bold',
                         bbox=dict(facecolor='red', alpha=0.7, edgecolor='none'))
        axes[2].set_title(f"Result: {len(defects)} Defects Verified"); axes[2].axis('off')
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def generate_text_analysis(self, defects):
        if not defects: return "✓ 검사 결과: 정상 (미세 영역 포함 특이사항 없음)"
        res = f"총 {len(defects)}건의 외곽/표면 결함 분석:\n"
        for i, d in enumerate(defects, 1):
            res += f"{i}. {d.clock_position} 방향 - {d.defect_type} (심각도: {d.severity:.2f})\n"
        return res