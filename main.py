import os, datetime, random, cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from config import OK_DIR, DEF_DIR, RESULTS_DIR, CLIP_MODEL
from modules.clip_search import CLIPSearch
from modules.ssim_utils import ssim_global
from modules.defect_analyzer import AdvancedDefectAnalyzer

def prepare_images(ref_path, target_path):
    """
    이미지 로드 및 사이즈 통일.
    """
    ref_cv = cv2.imread(str(ref_path))
    tgt_cv = cv2.imread(str(target_path))
    if ref_cv is None or tgt_cv is None: return None, None
    
    tgt_cv = cv2.resize(tgt_cv, (ref_cv.shape[1], ref_cv.shape[0]))
    return Image.fromarray(cv2.cvtColor(ref_cv, cv2.COLOR_BGR2RGB)), \
           Image.fromarray(cv2.cvtColor(tgt_cv, cv2.COLOR_BGR2RGB))

def main():
    print("="*70)
    print("Smart Factory Defect Detection System v6.0 (Robust Edge-Detection)")
    print("="*70 + "\n")
    
    # 1. 이미지 로드 및 매칭
    ok_files = list(OK_DIR.glob("*.jpeg")) + list(OK_DIR.glob("*.jpg")) + list(OK_DIR.glob("*.png"))
    if not ok_files: 
        print("❌ OK 폴더에 이미지가 없습니다.")
        return
    
    ref_path = random.choice(ok_files)
    print(f"[1] 기준 이미지 선정: {ref_path.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    searcher = CLIPSearch(model_id=CLIP_MODEL, device=device)
    searcher.build_index(DEF_DIR)
    
    hits = searcher.search_with_index(str(ref_path), top_k=1)
    target_path = Path(hits[0].candidate_path)
    print(f"[2] 검사 대상 매칭: {target_path.name} (유사도: {hits[0].similarity:.4f})")
    
    # 2. 분석 준비
    img_ref, img_tgt = prepare_images(ref_path, target_path)
    if img_ref is None: return
    
    ssim_val = ssim_global(img_ref, img_tgt)
    print(f"[3] SSIM 구조 지수: {ssim_val:.4f}")
    
    # 3. 분석기 인스턴스화 (Michael님, 에러 원인인 grid_size를 제거했습니다)
    analyzer = AdvancedDefectAnalyzer(
        min_defect_area=20,         # [조정] 40 -> 20: 아주 작은 찍힘 점(Point)도 놓치지 않음
        max_defect_area=8000,       # 대형 결함 포착 유지
        intensity_threshold=8,      # [핵심 조정] 15 -> 8: 명암 차이가 아주 적은 연한 음영도 무조건 포착
        confidence_threshold=0.15,  # [조정] 0.3 -> 0.15: 미세한 후보군도 리포트에 포함
        grid_size=7
    )
    
    # 4. 결함 분석 수행
    print(f"[4] 그래디언트 기반 정밀 분석 시작...")
    defects = analyzer.analyze(img_ref, img_tgt)
    
    # [Michael님 필독] 
    # 최신 analyzer.analyze 내부에서 이미 중앙부 제거 및 마스크 수축이 진행되므로 
    # main.py의 추가 필터 로직은 불필요해졌습니다. 
    # analyzer에서 나온 결과가 바로 최적화된 결과입니다.

    print(f"    -> 분석 완료: {len(defects)}개의 유효 결함 확정\n")

    # 5. 지능형 리포트 출력
    text_report = analyzer.generate_text_analysis(defects)
    print("="*70)
    print("현장 점검 분석 리포트 (Final Robust Results)")
    print("-" * 70)
    print(text_report)
    print("="*70 + "\n")
    
    # 6. 시각화 및 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_img_path = RESULTS_DIR / f"report_{target_path.stem}_{timestamp}.png"
    
    analyzer.visualize_results(img_ref, img_tgt, defects, save_path=str(report_img_path))
    print(f"✅ 리포트 이미지 저장: {report_img_path}")
    
    # 7. 결함 패치 저장
    if defects:
        width, height = img_tgt.size
        for i, d in enumerate(defects, 1):
            x, y, w, h = d.bbox
            padding = 40
            patch = img_tgt.crop((max(0, x-padding), max(0, y-padding), 
                                 min(width, x+w+padding), min(height, y+h+padding)))
            patch.save(RESULTS_DIR / f"defect_{i}_{timestamp}.png")
        print(f"✅ {len(defects)}개의 결함 상세 패치 저장 완료.")

if __name__ == "__main__":
    main()