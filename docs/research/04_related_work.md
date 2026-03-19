# 관련 연구 정리

## 1. Training-Free 탐지 방법 (25+ 방법, 7개 카테고리)

| 카테고리 | 대표 방법 | 핵심 메커니즘 | PhaseForensics와의 관계 |
|---------|----------|-------------|----------------------|
| Perturbation-Sensitivity | RIGID, WaRPAD, TRIM | DINOv2/CLIP 임베딩의 perturbation 민감도 | 직교적 (우리는 위상, 이들은 FM 반응) |
| Reconstruction-Error | AEROBLADE, DIRE, FIRE, R2BD | LDM VAE/diffusion reconstruction error | 직교적 (우리는 model-free) |
| Denoising Trajectory | Denoising Trajectory Biases, LATTE | Diffusion inversion 궤적 | 직교적 (우리는 diffusion 불필요) |
| **Spectral/Frequency** | **SPN, SPAI** | **FFT/DCT amplitude 스펙트럼** | **직접 경쟁 — amplitude만 사용, phase 미사용** |
| FM Probing | UnivFD, C2P-CLIP, FatFormer, AIDE | CLIP/DINOv2 feature + classifier | 직교적 (우리는 FM 불필요) |
| Self-Supervised | Forensic Self-Descriptions, CINEMAE | Real-only 학습 | 직교적 (우리는 학습 없음) |
| Test-Time Adaptation | TTP-AP, B-Free | Inference 시 적응 | 직교적 |

### 핵심 주파수 기반 베이스라인
- **SPN (Spectral Power Normalization)**: FFT amplitude의 radial average profile → GAN spectral peak 탐지
- **SPAI**: DCT coefficient 분포의 artifact 패턴
- **NPR**: 인접 픽셀 관계의 주파수 특성

**공통 한계**: 모두 **amplitude/magnitude** 스펙트럼만 분석. **Phase를 무시.**

## 2. Phase-based 이미지 분석 (비 forensics)

| 연구 | 분야 | 핵심 |
|------|------|------|
| Oppenheim & Lim (1981) | 신호처리 | Phase가 amplitude보다 이미지 구조에 중요 |
| Portilla & Simoncelli (2000) | 텍스처 합성 | 자연 이미지의 위상 상관 구조 |
| Local Phase Quantization (LPQ) | 텍스처 분류 | DFT local phase → blur-invariant descriptor |
| Phase congruency (Kovesi, 1999) | 에지 검출 | 다중 스케일 위상 일치 → 에지 |

**PhaseForensics의 차별점**: 위 연구들은 phase를 인식/분류에 사용. 우리는 **phase coherence violation을 forensic signal로** 사용하는 최초 연구.

## 3. Wavelet-based Forensics

| 연구 | 방법 | 한계 |
|------|------|------|
| Fridrich & Kodovsky (2012) | Steganalysis with SRM | 고정 필터, DL 이전 |
| Bayar & Stamm (2016) | Constrained CNN | 학습 기반 |
| Li et al. (2021) | Wavelet-based frequency analysis | Amplitude만 사용 |
| Frank et al. (2020) | Frequency-based GAN detection | DCT amplitude |

**공통 한계**: Wavelet을 사용하더라도 **amplitude/magnitude coefficient**만 분석. **Complex coefficient의 phase 정보를 forensics에 활용한 연구는 전무.**

## 4. 주요 참고 문헌 (논문에 인용할 것)

### Core references
1. Oppenheim, A. V., & Lim, J. S. (1981). The importance of phase in signals. *Proceedings of the IEEE*.
2. Selesnick, I. W., Baraniuk, R. G., & Kingsbury, N. C. (2005). The dual-tree complex wavelet transform. *IEEE Signal Processing Magazine*.
3. Portilla, J., & Simoncelli, E. P. (2000). A parametric texture model based on joint statistics of complex wavelet coefficients. *IJCV*.

### Forensics baselines
4. Ricker, J., et al. (2024). AEROBLADE: Training-free detection of latent diffusion images. *CVPR*.
5. Wang, Z., et al. (2023). DIRE: Diffusion-generated image detection. *ICCV*.
6. Ojha, U., et al. (2023). Towards universal fake image detectors (UnivFD). *CVPR*.
7. Koutlis, C., & Papadopoulos, S. (2024). WaRPAD. *arXiv*.

### GGD / Statistical detection
8. Do, M. N., & Vetterli, M. (2002). Wavelet-based texture retrieval using generalized Gaussian density and Kullback-Leibler distance. *TIP*.
9. Sharifi, K., & Leon-Garcia, A. (1995). Estimation of shape parameter for generalized Gaussian distributions. *IEEE TSP*.
