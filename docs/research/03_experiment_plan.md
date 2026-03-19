# PhaseForensics 실험 계획

## 전체 타임라인 (8주)

```
Week:  1    2    3    4    5    6    7    8
      [Proto][GGD][=Benchmark=][Compress][Ablation][===Paper===]
       Pilot      GenImage     JPEG     DWT vs    논문 집필
       Study      CNNDet       WebP     DT-CWT    시각화
                  UnivFake     H.264    amp+phase  Localization
```

## Week 1: 프로토타입 + Pilot Study (GO/NO-GO)

### 1.1 DT-CWT 파이프라인 구현
- dtcwt 패키지 기반 위상 추출
- ISPC(Inter-Scale Phase Coherence) 계산 모듈
- CDPGC(Cross-Direction Phase Gradient Consistency) 계산 모듈
- 시각화: ISPC/CDPGC 맵 출력

### 1.2 Pilot Study (필수, GO/NO-GO 결정)
- 생성기별 100장씩 수집:
  - **GAN**: ProGAN, StyleGAN2, StyleGAN3
  - **Diffusion**: SD1.5, SDXL, SD3, FLUX
  - **Commercial**: Midjourney v5/v6, DALL-E 3
  - **Real**: ImageNet/COCO 100장
- 각 생성기별 ISPC/CDPGC 히스토그램 시각화
- Cohen's d effect size 계산
- GO/NO-GO 판정

### 1.3 성공 기준
- Cohen's d > 0.5 for at least 7/9 generators → GO

## Week 2: GGD 파라미터 사전 피팅

### 2.1 자연 영상 GGD 모델 구축
- ImageNet ILSVRC 2012 validation set (50,000장) 중 5,000장 랜덤 샘플링
- COCO 2017 validation set (5,000장) 중 2,000장 추가
- 총 7,000장에서 DT-CWT ISPC/CDPGC 계산
- GGD(α, β) MLE fitting
- Cross-validation으로 fitting 안정성 확인

### 2.2 Neyman-Pearson 검출 threshold 설정
- False Positive Rate(FPR) 제어: α = {0.01, 0.05, 0.10}
- Likelihood ratio threshold 계산

## Week 3-4: 벤치마크 실험

### 3.1 데이터셋
| 데이터셋 | 규모 | 생성기 | 위치 |
|---------|------|--------|------|
| GenImage | 1.3M | 8 generators | `/mnt/storage/datasets/DeepfakeDetection/Image/GenImage/` |
| CNNDetection | 720K | 11 generators | `/mnt/storage/datasets/DeepfakeDetection/Image/CNNDetection/` |
| UniversalFakeDetect | 다수 | 다수 | `/mnt/storage/datasets/DeepfakeDetection/Image/UniversalFakeDetect/` |
| Synthbuster | TBD | 다수 | 다운로드 필요 |

### 3.2 평가 지표
- AUROC (primary)
- Average Precision (AP)
- TPR @ FPR={1%, 5%}
- 생성기별 breakdown

### 3.3 베이스라인 비교
| 방법 | 유형 | 핵심 |
|------|------|------|
| SPN (Spectral Power Normalization) | TF, amplitude | FFT 진폭 스펙트럼 |
| SPAI | TF, amplitude | 스펙트럼 artifact |
| WaRPAD | TF, perturbation | DINOv2 perturbation 민감도 |
| RIGID | TF, perturbation | CLIP perturbation |
| AEROBLADE | TF, reconstruction | LDM VAE reconstruction |
| NPR (Neighboring Pixel Relationships) | TF, pixel | 인접 픽셀 관계 |
| UnivFD | Light, FM probe | CLIP linear probe |

## Week 5: 압축 강건성 실험

### 5.1 JPEG 압축
- Quality Factor: {95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 40, 30}
- 각 QF에서 AUROC 측정
- 핵심 비교: PhaseForensics(phase) vs SPN/SPAI(amplitude) 압축 degradation curve

### 5.2 기타 압축
- WebP: quality {90, 75, 50, 25}
- H.264: CRF {18, 23, 28, 35, 40} (프레임 추출 후 적용)
- H.265/HEVC: CRF {18, 23, 28, 35, 40}

### 5.3 소셜 미디어 시뮬레이션
- Twitter/X: resize 1024→ JPEG QF85
- Instagram: resize 1080→ JPEG QF75
- Facebook: resize 2048→ JPEG QF71
- Telegram: 원본 유지 (비압축 채널)

## Week 6: Ablation Study

### 6.1 웨이블릿 변환 비교
- DWT (Discrete Wavelet Transform) — baseline
- DT-CWT (Dual-Tree Complex WT) — 제안 방법
- Steerable Pyramid — 대안

### 6.2 DT-CWT 파라미터
- Decomposition level L: {3, 4, 5, 6, 7}
- Wavelet basis: 기본(near_sym_b, qshift_b) vs 대안

### 6.3 위상 feature 비교
- ISPC only
- CDPGC only
- ISPC + CDPGC (제안)
- 개별 스케일/방향 기여도

### 6.4 탐지 방법 비교
- GGD + Neyman-Pearson (제안)
- GMM (Gaussian Mixture Model)
- KDE (Kernel Density Estimation)
- Earth Mover's Distance

### 6.5 Amplitude + Phase 앙상블
- Phase only (제안)
- Amplitude only (baseline)
- Phase + Amplitude (가중 결합)
- 최적 결합 가중치 (Fisher Discriminant)

## Week 7-8: 논문 집필 + 추가 실험

### 7.1 Localization 실험
- ISPC/CDPGC 맵 기반 위조 영역 탐지
- FF++ pixel-level mask 대비 IoU/F1
- Threshold 선택 방법 (Otsu's method 등)

### 7.2 시각화
- 생성기별 ISPC/CDPGC 맵 시각화 (real vs GAN vs diffusion)
- GGD 파라미터 분포 scatter plot
- 압축 degradation curve (phase vs amplitude)
- t-SNE/UMAP of phase features

### 7.3 논문 구조 (초안)
1. Introduction: amplitude만 보던 한계, phase의 잠재력
2. Related Work: frequency-based detection, phase analysis in vision
3. Method: DT-CWT → ISPC/CDPGC → GGD → NP test
4. Experiments: benchmark, compression, ablation, localization
5. Analysis: 왜 phase가 다른가 (생성기 디코더 분석)
6. Conclusion
