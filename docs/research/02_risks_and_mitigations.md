# PhaseForensics 리스크 분석 및 대응 전략

## 리스크 1: 최신 생성기의 위상 일관성 수렴 (심각도: 높음)

### 문제
FLUX, SD3.5, Midjourney v6 등 최신 diffusion model이 점점 더 자연스러운 phase structure를 생성할 가능성.
특히 flow matching 기반 모델은 이전 세대보다 정교한 디코더를 사용.

### 검증 방법
**Pilot Study (Week 1 필수)**:
1. 각 생성기(SD1.5, SDXL, SD3, FLUX, Midjourney v5/v6, DALL-E 3)로 생성된 이미지 100장씩 수집
2. DT-CWT ISPC/CDPGC 히스토그램을 생성기별로 그려서 real과의 분리도 확인
3. Cohen's d (effect size) 계산: d > 0.5이면 진행, d < 0.3이면 방향 전환

### 대응
- 최신 생성기에서 signal 약화 시: amplitude+phase 결합으로 보완
- Signal이 완전 소멸 시: FoMFP 또는 FAP로 전환 (보류 후보)

## 리스크 2: 소셜 미디어 Re-encoding (심각도: 중간)

### 문제
소셜 미디어(Twitter/X, Instagram, Facebook) 업로드 시:
- Resize (원본보다 작은 해상도)
- JPEG 재압축 (QF 70-85 수준)
- Watermark 추가
- Metadata stripping

이러한 복합 처리가 위상 일관성을 파괴할 수 있음.

### 검증 방법
- 소셜 미디어 시뮬레이션 파이프라인 구축 (resize → JPEG → noise)
- 각 단계별 ISPC/CDPGC 변화 추적
- Real/fake 분리도의 degradation 측정

### 대응
- 위상이 amplitude보다 강건하다는 것은 기존 연구에서 부분적으로 지지됨
- 복합 처리 후에도 signal이 유지되는 frequency band/scale을 선별적으로 사용
- 최악의 경우: "비압축/경미한 압축 환경에서의 탐지"로 scope 축소

## 리스크 3: GGD 모델 적합도 (심각도: 낮음)

### 문제
ISPC/CDPGC 분포가 GGD로 잘 근사되지 않을 수 있음 (model mismatch).

### 검증 방법
- Kolmogorov-Smirnov test로 GGD fitting의 적합도 검증
- Q-Q plot 시각화

### 대응
- GGD 외 대안: GMM, Kernel Density Estimation
- Non-parametric 접근: histogram distance (Earth Mover's Distance)

## 리스크 4: dtcwt 패키지 성능 (심각도: 낮음)

### 문제
- dtcwt Python 구현이 batch processing에 최적화되지 않아 대량 이미지에서 느릴 수 있음
- GGD fitting에서 shape parameter 초기값에 따라 수렴 실패 edge case 가능

### 대응
- 대량 처리 시: multiprocessing + 이미지 단위 병렬화
- GGD fitting: scipy.stats.gennorm 사용, robust 초기값 설정
- 필요 시 Cython wrapper 또는 PyWavelets 대안 검토

## 리스크 5: 논문 story의 설득력 (심각도: 중간)

### 문제
"위상 비일관성이 왜 생성 모델에서 발생하는가"를 아키텍처 수준에서 설명해야 함.
리뷰어가 "왜 반드시 그래야 하는가"라는 이론적 필연성을 요구할 수 있음.

### 대응
- Transposed convolution의 phase shift 분석 (checkerboard artifact와의 연결)
- Decoder upsampling의 phase randomization 시각화
- 생성기별 디코더 아키텍처 분석 → 위상 비일관성의 원인 매핑
- 교수 리뷰어 지적: "theoretical justification" 보강 필수

## GO/NO-GO 기준 (Week 1 Pilot Study 후)

| 조건 | 판정 |
|------|------|
| 최신 생성기 포함 Cohen's d > 0.5 | **GO** — 전체 실험 진행 |
| Cohen's d = 0.3~0.5 | **CONDITIONAL GO** — amplitude+phase 결합으로 진행 |
| Cohen's d < 0.3 | **NO-GO** — FoMFP 또는 FAP로 전환 |
| JPEG QF50에서 d > 0.3 | 압축 강건성 claim 가능 |
| JPEG QF50에서 d < 0.2 | 압축 강건성 claim 철회, 비압축만 |
