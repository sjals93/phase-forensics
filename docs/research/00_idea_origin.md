# PhaseForensics 아이디어 기원 및 근거

## 출처
- **dl_papers 프로젝트**: NeurIPS/CVPR/ICLR 2020-2025 총 50,348편 논문 자동 분석
- **Phase 1**: 8개 병렬 에이전트로 13개 후보 생성 → 10개 선별
- **Phase 2**: 3인 검수단 (교수/엔지니어/DL고수) 독립 평가
- **최종 결과**: PhaseForensics가 종합 1위 (37.3/50, 작동확률 65%)

## 왜 PhaseForensics인가?

### 3인 검수단 평가 요약

| 검수자 | 점수 | 핵심 평가 |
|--------|:----:|----------|
| 교수 | 33/50 | 위상이라는 미개척 탐지 신호, 압축 강건성 강점. 최신 생성기 위상 수렴 가능성 우려 |
| 엔지니어 | 43/50 | **전체 10개 중 최고점**. 구현 최소, CPU 실행, dtcwt만 있으면 즉시 시작 |
| DL 고수 | 36/50 | 물리적 기반 sound, 작동확률 65%. 최신 생성기 pilot study 필수 |

### 경쟁 후보 대비 강점
1. **구현 난이도 최저**: 0.5주 프로토타입 가능 (vs FoMFP 2주, SAFE-Conformal 4주)
2. **GPU 불필요**: CPU에서도 실시간 → A6000을 다른 실험에 동시 사용 가능
3. **완전 Training-Free**: 위조 데이터 학습 전무, 자연 영상 GGD만 사전 피팅
4. **압축 강건성**: 위상은 amplitude보다 JPEG에 강건 (Oppenheim & Lim, 1981)
5. **3인 전원 상위권 합의**: 최저 33, 최고 43. 의견 불일치 최소

### 기대 성능
- 비압축 zero-shot: AUROC 85-90%
- **강압축 QF50: 진폭 기반 대비 AUROC +5-15% (핵심 selling point)**
- Cross-generator zero-shot: AUROC 85-90%

### 논문 story
"기존 주파수 방법은 amplitude만 보았다. Phase에는 더 풍부하고 압축에 강건한 신호가 있다."

## 연구 갭 (50K 논문 분석에서 확인)
- Training-free 탐지의 10대 연구 갭 중:
  - **#7**: 해상도/압축에 robust한 TF 탐지 부재 → PhaseForensics가 직접 해결
  - **#4**: 이론적 근거 부재 → 물리적 원리(PSF + phase alignment)로 해결
  - **#6**: Multi-signal 융합 미시도 → amplitude+phase 앙상블로 확장 가능
