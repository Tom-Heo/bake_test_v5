import os


class Config:
    # -------------------------------------------------------------------------
    # [Path Settings]
    # -------------------------------------------------------------------------
    TRAIN_DATA_ROOT = os.path.join("dataset", "japan_test")

    # 체크포인트 및 로그 저장 경로
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULT_DIR = "results"

    # 마지막 학습 상태 자동 로드 파일명
    LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last.pth")

    # -------------------------------------------------------------------------
    # [Data Settings]
    # -------------------------------------------------------------------------
    # v5 모델의 Body 연산량과 96ch Loss 메모리 부하를 고려하여 안정적인 1로 설정
    BATCH_SIZE = 1
    ACCUM_STEPS = (
        16  # Gradient Accumulation: effective batch size = BATCH_SIZE * ACCUM_STEPS
    )
    NUM_WORKERS = 4  # 데이터 로더 워커 수

    # -------------------------------------------------------------------------
    # [Model Settings]
    # -------------------------------------------------------------------------
    INTERNAL_DIM = 30  # BakeNet 내부 연산 차원 (Internal Baking)
    EMA_DECAY = 0.999  # EMA 감쇠율

    # -------------------------------------------------------------------------
    # [Training Settings]
    # -------------------------------------------------------------------------
    TOTAL_EPOCHS = 10000  # 총 학습 에폭

    # Optimizer (AdamW) - v5의 안정적인 설정
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6  # 디테일 보존을 위해 약한 규제 적용

    # Scheduler (ExponentialLR)
    SCHEDULER_GAMMA = 0.999996  # 아주 천천히 떨어지는 학습률

    # 주기 설정
    LOG_INTERVAL_STEPS = 1  # N optimizer step마다 로그 출력
    VALID_INTERVAL_EPOCHS = 1  # 1 에폭마다 체크포인트 저장

    # -------------------------------------------------------------------------
    # [Hardware Settings]
    # -------------------------------------------------------------------------
    DEVICE = "cuda"  # CUDA 사용 필수
    USE_AMP = False  # 정밀도 유지를 위해 AMP(Mixed Precision) 미사용

    @classmethod
    def create_directories(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)
