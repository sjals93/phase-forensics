# PhaseForensics Project

## 프로젝트 개요
DT-CWT(Dual-Tree Complex Wavelet Transform) 위상 일관성(phase coherence)을 활용한 Training-Free 합성 이미지 탐지 시스템.

**정식 제목**: *PhaseForensics: Training-Free Synthetic Image Detection via Local Phase Coherence Violation in the Wavelet Domain*

**목표 학회**: CVPR 2027 / ECCV 2026

## 핵심 아이디어
- 기존 주파수 기반 탐지는 **진폭(amplitude)** 스펙트럼에만 의존
- 본 연구는 **위상(phase)** 일관성을 최초로 체계적으로 활용
- 실제 카메라 PSF → 스케일/방향 간 위상 정렬 생성
- 생성 모델 디코더의 독립적 업샘플링 → 위상 비일관성 발생
- 위상 정보는 압축에 상대적으로 강건 → **JPEG 압축 환경에서 핵심 강점**

## 환경
- GPU: NVIDIA RTX A6000 (48GB) — 단, 이 프로젝트는 CPU만으로도 실행 가능
- OS: Ubuntu 22.04
- Python 3.10+
- 데이터셋 경로: `/mnt/storage/datasets/`

## 코드 컨벤션
- Python, type hints 사용
- 실험 설정은 `configs/` 디렉토리의 YAML 파일로 관리
- 결과는 `results/` 디렉토리에 실험명/날짜별로 저장
- 로그는 `logging` 모듈 사용 (print 금지)

## 데이터셋 위치
```
/mnt/storage/datasets/
├── DeepfakeDetection/
│   ├── Image/          (GenImage, CNNDetection, UniversalFakeDetect)
│   └── AI_Generated/   (Stable Diffusion, Midjourney outputs)
└── GeneralDatasets/    (COCO, ImageNet subset for GGD fitting)
```

## 이메일 알림
- Gmail: sjals93@gmail.com
- App Password: ibja bxch bcmr cith
- 장시간 실험 완료 시 이메일 알림 발송

## 언어
- 사용자는 한국어로 소통. 코드 주석/변수명은 영어.
