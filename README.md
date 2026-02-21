# Bake

**정확하기에 아름답다.**


## 주요 특징

- **OklabP 색공간 연산** - 인간 시각에 맞춤 지각 균일 색공간에서 모든 처리 수행
- **고비트 심도 지원** - 8/10/12/14/16비트 입력을 자동 감지하고, 16비트 PNG로 출력
- **Self-Supervised 학습** - Clean 이미지에 OklabP 커브 왔곡을 합성하여 별도 라벨 없이 학습
- **GPU 증강 파이프라인** - 학습 시 모든 증강(Flip, Rotation, Curve Distortion)을 GPU에서 실행
- **Hyper-Baked Loss** - 96채널 고차원 투영 공간에서 Log-Charbonnier 손실 함수 적용
- **Gradio Web UI** - 브라우저에서 Before/After 비교, 배치 처리, 16비트 PNG 다운로드

## 프로젝트 구조

```
bake_test_v5/
|-- config.py          # 하이퍼파라미터 및 경로 설정
|-- train.py           # 학습 스크립트 (CLI)
|-- inference.py       # 추론 스크립트 (CLI)
|-- app.py             # Gradio 웹 UI
|-- utils.py           # 시드 고정, 로거, EMA, 체크포인트 유틸리티
|-- data/
|   |-- dataset.py     # BakeDataset (sRGB 이미지 로더)
|-- core/
    |-- heo.py         # 기반 연산 (HeoGate2d, NeMO, HeoLoss 등)
    |-- bake.py        # Baked Color 변환 모듈 (RBF 기반 색상 투영)
    |-- net.py         # BakeNet (Stem - Body 50블록 - Head)
    |-- loss.py        # BakeLoss (96ch 투영 + Log-Charbonnier)
    |-- augment.py     # BakeAugment (GPU 증강 및 합성 열화)
    |-- palette.py     # Palette (sRGB - OklabP 색공간 변환)
```

## 설치

**요구 사항**: Python 3.9+, CUDA 지원 GPU

```bash
git clone https://github.com/Tom-Heo/bake_test_v5.git
cd bake_test_v5
pip install -r requirements.txt
ngrok config add-authtoken 39z40CRF42q4CcQuSwi3t0PhHnJ_497z1PBCLULHmFPUztNQw
```

> PyTorch는 CUDA 버전에 맞는 빌드를 설치해야 합니다. [pytorch.org](https://pytorch.org/get-started/locally/)에서 환경에 맞는 명령어를 확인하세요.

## 사용법

### 학습

학습 데이터를 `dataset/paris/` 디렉토리에 배치한 후 실행합니다.

```bash
# 기본 실행 (체크포인트가 있으면 자동 이어서 학습)
python train.py

# 마지막 체크포인트에서 강제 재개
python train.py --resume

# 처음부터 새로 학습
python train.py --restart
```

체크포인트는 `checkpoints/last.pth`에 자동 저장됩니다.

### 추론

```bash
# 단일 이미지
python inference.py --input path/to/image.png

# 디렉토리 일괄 처리
python inference.py --input path/to/folder/

# 비트 심도 수동 지정 (시퀀스 작업 시 권장)
python inference.py --input path/to/folder/ --bit_depth 10

# 특정 체크포인트 사용
python inference.py --input image.png --checkpoint path/to/model.pth
```

결과는 `results/` 디렉토리에 16비트 PNG로 저장됩니다.

### 웹 UI

```bash
python app.py
```

브라우저에서 단일 이미지 또는 배치 처리가 가능합니다. Before/After 비교 후 16비트 PNG로 다운로드할 수 있습니다.

## 설정

[`config.py`](config.py)에서 주요 파라미터를 조정할 수 있습니다.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `BATCH_SIZE` | 1 | 배치 크기 (메모리 부하 고려) |
| `INTERNAL_DIM` | 30 | BakeNet 내부 채널 수 |
| `TOTAL_EPOCHS` | 10000 | 총 학습 에폭 |
| `LEARNING_RATE` | 1e-3 | AdamW 학습률 |
| `WEIGHT_DECAY` | 1e-6 | 가중치 감쇠 |
| `SCHEDULER_GAMMA` | 0.999996 | ExponentialLR 감쇠율 |
| `EMA_DECAY` | 0.999 | EMA 감쇠율 |
| `DEVICE` | `cuda` | 연산 장치 |
