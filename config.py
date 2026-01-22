from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OK_DIR = DATA_DIR / "ok"
DEF_DIR = DATA_DIR / "def"
RESULTS_DIR = BASE_DIR / "results"

# 모델 설정
CLIP_MODEL = "ViT-B-32/openai"

# 분석 임계값 (결과 판정용)
SSIM_THRESHOLD = 0.90
DENT_THRESHOLD = 0.02

# 폴더 자동 생성
for d in [OK_DIR, DEF_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)