import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from typing import List, Tuple

class ImageProcessor:
    """향상된 이미지 처리 및 시각화 유틸리티"""
    
    @staticmethod
    def create_heatmap_overlay(img: np.ndarray, mask: np.ndarray, alpha=0.6) -> np.ndarray:
        """
        결함 영역을 히트맵으로 시각화
        """
        # 컬러맵 적용
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        # 원본 이미지와 블렌딩
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
        
        return overlay
    
    @staticmethod
    def draw_clock_diagram(ax, defects: List, img_shape: Tuple[int, int]):
        """
        12시간 다이어그램 표시 (결함 위치 시각화)
        """
        h, w = img_shape[:2]
        center_x, center_y = w / 2, h / 2
        
        # 12시간 눈금 그리기
        for hour in range(1, 13):
            angle = (hour - 3) * 30  # 3시 = 0도 기준
            rad = np.radians(angle)
            
            # 선 그리기
            r1, r2 = min(w, h) * 0.35, min(w, h) * 0.4
            x1 = center_x + r1 * np.cos(rad)
            y1 = center_y - r1 * np.sin(rad)
            x2 = center_x + r2 * np.cos(rad)
            y2 = center_y - r2 * np.sin(rad)
            
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.5)
            
            # 시간 표시
            r3 = min(w, h) * 0.45
            tx = center_x + r3 * np.cos(rad)
            ty = center_y - r3 * np.sin(rad)
            ax.text(tx, ty, str(hour), ha='center', va='center', 
                   fontsize=10, color='yellow', fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
        
        # 결함 위치 마커
        for defect in defects:
            cx, cy = defect.center_coords
            ax.plot([center_x, cx], [center_y, cy], 'r--', linewidth=2, alpha=0.8)
            ax.scatter(cx, cy, c='red', s=100, marker='X', zorder=10, 
                      edgecolors='white', linewidths=2)
    
    @staticmethod
    def show_analysis_pipeline(ref_path, tgt_path, defects: List, 
                               obj_mask: np.ndarray, defect_map: np.ndarray,
                               save_path: str = None):
        """
        전체 분석 파이프라인 시각화 (5단계)
        """
        # 이미지 로드
        img_ref = cv2.imread(str(ref_path))
        img_tgt = cv2.imread(str(tgt_path))
        img_tgt = cv2.resize(img_tgt, (img_ref.shape[1], img_ref.shape[0]))
        
        # RGB 변환
        img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        img_tgt_rgb = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2RGB)
        
        # Figure 생성
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.2)
        
        # 1. 정상 이미지
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_ref_rgb)
        ax1.set_title("① Standard Reference", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. 검사 대상
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_tgt_rgb)
        ax2.set_title("② Target Image", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. 객체 분할
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = img_tgt_rgb.copy()
        overlay[obj_mask == 0] = overlay[obj_mask == 0] * 0.3
        ax3.imshow(overlay)
        ax3.set_title("③ Object Segmentation", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. 결함 맵 (히트맵)
        ax4 = fig.add_subplot(gs[0, 3])
        heatmap = ImageProcessor.create_heatmap_overlay(img_tgt, defect_map, alpha=0.7)
        ax4.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        ax4.set_title("④ Defect Heatmap", fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. 최종 검출 결과 (시계 다이어그램 포함)
        ax5 = fig.add_subplot(gs[:, 4])
        ax5.imshow(img_tgt_rgb)
        
        # 시계 다이어그램
        ImageProcessor.draw_clock_diagram(ax5, defects, img_tgt.shape)
        
        # 결함 바운딩 박스
        for i, defect in enumerate(defects, 1):
            x, y, w, h = defect.bbox
            
            # 심각도별 색상
            if defect.severity > 0.7:
                color, style = 'red', '-'
            elif defect.severity > 0.4:
                color, style = 'orange', '--'
            else:
                color, style = 'yellow', ':'
            
            # 바운딩 박스
            rect = FancyBboxPatch((x, y), w, h, linewidth=3, 
                                 edgecolor=color, facecolor='none',
                                 linestyle=style, boxstyle='round,pad=2')
            ax5.add_patch(rect)
            
            # 라벨
            label = f"{i}"
            ax5.text(x, y-5, label, color='white', fontsize=14, 
                    bbox=dict(boxstyle='circle', facecolor=color, alpha=0.9),
                    ha='center', fontweight='bold')
        
        ax5.set_title(f"⑤ Final Detection ({len(defects)} Defects)", 
                     fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # 하단 상세 정보 패널
        info_ax = fig.add_subplot(gs[1, :3])
        info_ax.axis('off')
        
        # 결함 정보 테이블
        if defects:
            table_data = [["No.", "위치", "유형", "심각도", "깊이(mm)", "신뢰도"]]
            for i, d in enumerate(defects, 1):
                table_data.append([
                    str(i),
                    d.clock_position,
                    d.defect_type,
                    f"{d.severity:.2f}",
                    f"{d.estimated_depth:.2f}",
                    f"{d.confidence:.1%}"
                ])
            
            table = info_ax.table(cellText=table_data, cellLoc='center', loc='center',
                                 colWidths=[0.1, 0.25, 0.2, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # 헤더 스타일
            for i in range(6):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 데이터 행 스타일
            for i in range(1, len(defects)+1):
                severity = defects[i-1].severity
                if severity > 0.7:
                    bg_color = '#ffcdd2'
                elif severity > 0.4:
                    bg_color = '#fff9c4'
                else:
                    bg_color = '#e8f5e9'
                
                for j in range(6):
                    table[(i, j)].set_facecolor(bg_color)
        else:
            info_ax.text(0.5, 0.5, '✅ 결함 없음 (PASS)', 
                        ha='center', va='center', fontsize=20, 
                        color='green', fontweight='bold')
        
        plt.suptitle("Smart Factory Defect Analysis Report", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    @staticmethod
    def show_pairs(ok_path, def_path, similarity, save_path=None):
        """
        레거시 호환용 간단한 시각화 (기존 코드와의 호환성 유지)
        """
        img_ok = cv2.imread(str(ok_path))
        img_def = cv2.imread(str(def_path))
        img_def = cv2.resize(img_def, (img_ok.shape[1], img_ok.shape[0]))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        axes[0].imshow(cv2.cvtColor(img_ok, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Standard (OK)", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Target (Similarity: {similarity:.3f})", fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()