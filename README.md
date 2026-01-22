# ■ 유사 이미지 검색 및 차이점 검출 모델 개발

<br>

## 1. Project Overview
제조 현장에서 설비 및 제품의 불량 발생 시, AI를 통해 유사 불량 사례 검색, 정밀 결함 검출, 그리고 LLM 기반 대응 매뉴얼을 실시간으로 제공하는 지능형 품질 관리 솔루션입니다. 품질 관리자의 숙련도에 의존하던 기존 판단 과정을 자동화하여 대응 속도와 정확도를 혁신적으로 향상시킵니다.

<br>
<img width="760" height="283" alt="image" src="https://github.com/user-attachments/assets/14c1edd4-221d-4dc4-b4ab-2881656b1194" />


## 2. Key Objectives
1) 유사 사례 검색: 업로드된 이미지와 가장 유사한 과거 불량 사례 및 매뉴얼 매칭
2) 정밀 결함 검출: 기준(OK) 이미지와 검사 이미지 간의 미세 차이점(찍힘, 스크래치 등) 자동 식별
3) 지능형 가이드 생성: 검출된 결함의 원인을 분석하고 작업자를 위한 실시간 조치 가이드 생성

<br>
## 3. Technical Architecture & Strategy
1) 유사 이미지 검색 (VLM ViT-B-32/openai)
- Model Selection: CLIP (Contrastive Language-Image Pre-training)
- Approach: VLM 모델을 활용하여 이미지의 고차원 특징 벡터를 추출한 뒤, Cosine Similarity를 기반으로 벡터 데이터베이스에서 가장 유사한 과거 불량 사례를 검색합니다.
- Reasoning: 단순 픽셀 비교가 아닌 문맥적 의미(Contextual Semantic)를 파악하여 현장의 다양한 조명 및 각도 변화에도 강건한 검색 성능을 확보하기 위함입니다. <br>

2) 결함 분석 로직의 진화 (Hybrid Approach)
- VLM의 한계점 (Challenge): VLM은 이미지 캡셔닝과 객체 명칭 인식에는 뛰어나나, 산업 현장에서 요구하는 **정밀한 세그멘테이션(Segmentation)**과 미세한 물리적 특징(찌그러짐, 스크래치 등) 분석에는 한계가 있었습니다. 특히 텍스트 변환 과정에서 미세한 기하학적 변형 정보가 손실되는 문제가 발생했습니다.
- 해결 방안 (Innovation - Michael's Logic): VLM의 한계를 극복하기 위해 CNN/RNN/LSTM 응용 로직을 결합한 하이브리드 특징 분석 엔진을 자체 구현하였습니다.
- 다중 구역 분할 분석 (Region-based Analysis): 전체 이미지를 총 3가지 주요 구역으로 분할하여 구역별 특성에 최적화된 검사 수행.
- 민감도 제어 시스템: 구역별로 민감도를 다르게 설정하여 외곽부의 미세 찍힘과 중앙부의 구조적 결함을 분리하여 정밀 분석.
- 특징 추출 최적화: 이미지의 음영 변화와 픽셀 시퀀스의 연속성을 RNN/LSTM 구조로 해석하여, 단순 노이즈가 아닌 실제 물리적 변형(스크래치 등)을 시간적/공간적 흐름으로 파악.<br>
