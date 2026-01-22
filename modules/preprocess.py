from PIL import Image, ImageEnhance

def apply_preprocess(img: Image.Image) -> Image.Image:
    """나중에 이미지 전처리가 필요할 때 이 함수를 수정하세요."""
    # 예: 대비를 1.2배 높이고 싶다면 아래 주석을 해제
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(1.2)
    return img