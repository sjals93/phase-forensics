# DL 고수 검수 평가 — PhaseForensics

**점수: 36/50, 실제 작동 확률: 65%**

## 세부 평가

### 방법론 건전성: 8/10
DT-CWT phase coherence는 수학적으로 잘 정의되어 있고, amplitude 대비 phase의 강건성은 signal processing 분야에서 오래 알려진 사실.
GGD fitting + Neyman-Pearson은 classical하지만 solid한 접근.

### 신호 강도: 7/10
GAN, diffusion 모두 주로 amplitude spectrum을 "잘" 맞추는 데 집중하고 phase coherence는 간과하는 경향.
특히 inter-scale phase coherence는 자연 이미지의 강한 통계적 규칙성이므로 위조 이미지에서 깨질 가능성이 높음.

### SOTA 차별성: 7/10
대부분의 frequency-based 방법이 FFT amplitude만 보는데, phase 도메인으로의 전환은 명확한 차별화.
DT-CWT는 shift-invariant이므로 DFT보다 우수.

### 강건성: 7/10
Phase 정보가 amplitude보다 compression에 강건하다는 가설은 부분적으로 지지됨 (Oppenheim & Lim, 1981).
다만 강한 JPEG quantization은 phase도 파괴. 중간 QF(50-70)에서는 유리할 수 있으나 극단적 압축에서는 한계.

### 확장성/일반화: 7/10
Phase coherence는 generative process-agnostic한 특성이므로 새 생성기에도 일반화 가능성 높음.
비디오 확장 시 temporal phase coherence도 추가 가능.

## 핵심 의문
"최신 diffusion model(FLUX, SD3 등)이 phase coherence까지 잘 재현하는 수준에 이르렀는가?"
만약 최신 모델들이 이미 phase-aware 생성을 한다면 signal이 약해짐.
**Pilot study에서 생성기별 phase coherence 분포를 반드시 확인해야 함.**

## 숨겨진 가정 (Hidden Assumptions)
1. 최신 생성 모델(FLUX, SD3 등)이 여전히 phase coherence를 위반한다
2. DT-CWT의 decomposition level과 방향 수가 충분히 세밀하다
3. GGD가 ISPC/CDPGC 분포를 잘 근사한다
4. 자연 영상의 위상 통계가 카메라/해상도에 걸쳐 안정적이다

## "진짜 될 것" 분류: TOP 3 (3위)
물리적 기반이 sound하므로 부분적으로는 작동할 것.
단, 최신 생성기에서의 효과 감소 가능성이 주요 불확실성.
