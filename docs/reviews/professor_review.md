# 교수 검수 평가 — PhaseForensics

**점수: 33/50**

## 세부 평가

### 노벨티: 7/10
위상(phase) 일관성을 체계적으로 활용하는 것은 주파수 기반 forensics에서 의미 있는 gap을 메움.
기존이 amplitude에 치중했다는 지적은 정당하며, DT-CWT의 phase coherence를 forensic signal로 정식화하는 것은 새로움.
다만 phase-based 이미지 분석은 texture analysis 분야에서 이미 성숙한 도구(Local Phase Quantization 등).

### 이론적 깊이: 7/10
카메라 PSF의 convolution이 phase alignment을 생성한다는 물리적 근거가 있으며, GGD fitting + Neyman-Pearson framework은 통계적으로 원칙적.
DT-CWT의 shift-invariance 특성도 이론적으로 well-motivated.
다만 "생성 모델이 위상 비일관성을 남긴다"는 가정이 모든 생성 아키텍처에 성립하는지 이론적 보장 없음.

### 학술적 임팩트: 6/10
Amplitude vs Phase의 상보성을 보여주면 기존 주파수 방법에 즉시 통합 가능하여 실용적 임팩트는 있으나,
새로운 연구 방향을 개척하기보다는 기존 방향의 완성에 가까움.

### Accept 가능성: 6/10
깔끔한 실험 + 압축 강건성에서 명확한 우위를 보이면 accept 가능.
하지만 최신 생성 모델이 phase coherence를 학습하기 시작하면 방법의 수명이 짧을 수 있다는 리뷰어 우려 예상.

### 치명적 결함: 있음(관리 가능)
1. 최신 diffusion model (특히 flow matching 기반)이 점점 더 자연스러운 phase structure를 생성할 가능성. 방법의 temporal validity가 불확실.
2. 소셜 미디어 re-encoding (다중 압축)에서 원래의 phase coherence도 파괴될 수 있어 real/fake 구분력 약화.

## 탑티어 accept을 위한 보완 사항
1. "왜 생성 모델이 phase를 정확히 재현하지 못하는가"를 아키텍처 분석으로 설명
   - transposed convolution의 phase shift
   - decoder upsampling의 phase randomization
2. Phase + Amplitude 결합의 이론적 최적 가중치 도출
3. 최신 flow matching 모델에서의 phase coherence 측정 → 방법의 미래 유효성 평가

## 추천 학회
CVPR (실험 중심, 물리적 직관이 vision 커뮤니티에 어필)
