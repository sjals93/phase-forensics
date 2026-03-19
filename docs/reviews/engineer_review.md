# 엔지니어 검수 평가 — PhaseForensics

**점수: 43/50 (전체 10개 아이디어 중 최고)**

## 세부 평가

### 구현 실현성: 9/10
`dtcwt` 패키지는 pip install 한 줄. DT-CWT 위상 추출 + GGD fitting + Neyman-Pearson test는 모두 NumPy/SciPy 수준 구현. GPU 불필요라 환경 의존성 최소.

### 실험 재현성: 9/10
DT-CWT는 deterministic. GGD fitting은 MLE로 유일해. Neyman-Pearson threshold는 calibration set에서 결정되므로 set만 고정하면 완벽 재현.

### 계산 효율: 10/10
CPU에서도 실시간 가능. 이미지당 수십 ms. GPU 불필요하므로 edge device 배포도 가능. 메모리 사용량 무시 가능 수준.

### 공학적 건전성: 7/10
GGD assumption이 모든 생성기 유형에 대해 성립하는지 검증 필요. Neyman-Pearson은 binary hypothesis test이므로 multi-class 확장이 자연스럽지 않음.

### 실용적 가치: 8/10
CPU 실시간, 경량, 해석 가능. Edge/모바일 배포 이상적. 다른 방법의 보조 feature로 활용 시 가치 극대화.

## Engineering Traps (주의사항)

### 1. dtcwt batch processing
`dtcwt` 패키지의 Python 구현이 batch processing에 최적화되어 있지 않음.
대량 이미지 처리 시 for-loop이 되어 throughput 급감.
**대응**: multiprocessing Pool, 또는 PyWavelets 대안 검토.

### 2. GGD fitting edge case
GGD fitting에서 shape parameter 초기값에 따라 수렴 실패하는 edge case 존재.
**대응**: scipy.stats.gennorm.fit() 사용, robust 초기값(β=1.0), 수렴 실패 시 fallback (Laplacian β=1).

### 3. Phase wrapping
위상값이 [-π, π] 범위로 wrapping되므로 위상 차이 계산 시 circular statistics 필요.
**대응**: `np.angle(c1 * np.conj(c2))` 사용하여 자동 unwrapping.

## 구현 예상 시간: 0.5주 (프로토타입)
## 전체 실험 + 논문: 6-8주
