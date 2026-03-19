# PhaseForensics 방법론 상세

## 핵심 컨셉

### 물리적 원리
1. **실제 카메라**: PSF(Point Spread Function) 기반 컨볼루션 → 스케일/방향 간 위상의 체계적 정렬 생성
2. **생성 모델**: 디코더의 독립적 업샘플링(transposed convolution, pixel shuffle) → 위상 비일관성 발생
3. **핵심 관찰**: 위상 정보는 이미지의 구조적 정보를 인코딩하며, 압축에 상대적으로 강건

### 이론적 배경
- Oppenheim & Lim (1981): 위상이 amplitude보다 이미지 구조에 더 중요
- 자연 이미지의 local phase coherence는 잘 알려진 통계적 규칙성
- Portilla & Simoncelli (2000): 자연 이미지의 위상 상관 구조

## 4단계 파이프라인

### Step 1: DT-CWT 다중 스케일 위상 추출
```
Input Image
    ↓
DT-CWT (L=4~6 levels, 6 directions)
    ↓
Complex wavelet coefficients c_{l,d}(x,y)
    ↓
Phase extraction: φ_{l,d}(x,y) = arg(c_{l,d}(x,y))
```

**왜 DT-CWT인가?**
- 일반 DWT: shift-variant → 위상 추정 불안정
- DT-CWT: approximate shift-invariance → 안정적 위상 추정
- 6방향 분해 → 방향별 위상 분석 가능
- Python 패키지: `dtcwt` (pip install dtcwt)

### Step 2: 국소 위상 일관성 맵 계산 (2가지 핵심 지표)

#### (a) Inter-Scale Phase Coherence (ISPC)
인접 스케일 간 위상 차이의 원형 분산(circular variance):

```
ISPC(x,y,d) = 1 - |mean_l( exp(j * (φ_{l,d}(x,y) - φ_{l+1,d}(x,y))) )|
```

- Real 이미지: PSF의 컨볼루션으로 인해 인접 스케일 위상 차이가 일관적 → ISPC 낮음
- Fake 이미지: 독립적 업샘플링으로 스케일 간 위상 무관 → ISPC 높음

#### (b) Cross-Direction Phase Gradient Consistency (CDPGC)
같은 스케일, 다른 방향의 위상 기울기 잔차:

```
CDPGC(x,y,l) = Var_d( ∇φ_{l,d}(x,y) · e_d⊥ )
```

- Real 이미지: edge 위치에서 방향 간 위상 기울기가 일관적 (동일 edge가 여러 방향에서 동일 위상 전이 생성)
- Fake 이미지: 방향 간 위상 기울기 불일치

### Step 3: 통계적 탐지 점수화

#### GGD (Generalized Gaussian Distribution) Fitting
ISPC/CDPGC 히스토그램을 GGD로 피팅:

```
p(x; α, β) = (β / 2αΓ(1/β)) * exp(-(|x|/α)^β)
```

- α: scale parameter (분포의 폭)
- β: shape parameter (분포의 형태, β=2이면 가우시안, β=1이면 라플라시안)

**핵심 feature**: shape parameter β + tail probability P(X > threshold)

#### 자연 영상 GGD 사전 추정
- ImageNet/COCO에서 1000-5000장 real 이미지로 GGD 파라미터 사전 추정 (1회)
- 이것이 유일한 사전 계산. 위조 데이터는 전혀 필요 없음

#### Neyman-Pearson Likelihood Ratio Test
```
Λ(x) = p(x | H_fake, GGD_fake) / p(x | H_real, GGD_real)
```

- H_real의 GGD 파라미터: 사전 추정된 자연 영상 통계
- 테스트 이미지의 GGD 파라미터를 추정하여 H_real과의 편차를 점수화

### Step 4: Localization (위조 영역 자동 탐지)
- ISPC/CDPGC 맵 자체가 spatial resolution을 가지므로 위조 영역 localization 가능
- Threshold 기반 binary mask 생성 → IoU/F1 평가

## 확장 가능성

### 1. Amplitude + Phase 앙상블
- 기존 amplitude 기반 방법(SPN, FFT 스펙트럼 분석)과 결합
- 이론적 최적 가중치 도출 가능 (Fisher Linear Discriminant)

### 2. FoMFP의 freq view 대체
- PhaseForensics의 ISPC/CDPGC를 FoMFP의 LoRA_freq 입력으로 사용
- Phase signal → LoRA가 학습하기 더 좋은 feature 제공

### 3. SAFE-Conformal의 S_freq score
- PhaseForensics 점수를 SAFE-Conformal의 spectral score로 사용
- Conformal prediction으로 통계적 보장 추가

### 4. 비디오 확장
- 프레임별 위상 일관성 → temporal phase coherence
- 시간축 위상 변화의 일관성 분석
