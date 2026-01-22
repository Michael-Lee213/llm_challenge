import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def ssim_global(img1: Image.Image, img2: Image.Image) -> float:
    """두 이미지의 구조적 유사도(SSIM)를 계산합니다."""
    # 크기 통일
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # 그레이스케일 변환 및 배열화
    arr1 = np.array(img1.convert("L"))
    arr2 = np.array(img2.convert("L"))
    
    score, _ = ssim(arr1, arr2, full=True)
    return float(score)