from __future__ import annotations

import os
import random
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class BakeDataset(Dataset):
    """
    [Lightweight Dataset Loader]
    - CPU Heavy Task 제거 (열화는 GPU Augmentor로 이관)
    - 역할: Image Read -> Random Crop -> ToTensor
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root_dir = config.TRAIN_DATA_ROOT

        # 이미지 리스트 로드
        self.image_files = self._scan_files()

    def _scan_files(self) -> list[str]:
        """이미지 파일 목록 스캔"""
        files = sorted(
            [
                f
                for f in os.listdir(self.root_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        print(f"[Dataset] Loaded {len(files)} images from {self.root_dir}")
        return files

    def __len__(self):
        return len(self.image_files)

    def _get_scale_factor(self, w: int, h: int) -> int:
        long_side = max(w, h)
        if long_side >= 7680:
            return 6
        elif long_side >= 3840:
            return 3
        else:
            return 2

    def _get_random_crop(self, img: Image.Image, size=512) -> Image.Image:
        """
        [CPU-Side Crop]
        DataLoader가 배치를 구성하려면 모든 텐서의 크기가 같아야 하므로,
        여기서 1차적으로 512x512 크기로 잘라서 보냅니다.
        """
        w, h = img.size

        # 이미지가 Crop Size보다 작으면 Reflect Padding
        if w < size or h < size:
            pad_w = max(size - w, 0)
            pad_h = max(size - h, 0)
            img = TF.pad(img, (0, 0, pad_w, pad_h), padding_mode="reflect")
            w, h = img.size

        i = random.randint(0, h - size)
        j = random.randint(0, w - size)

        return TF.crop(img, i, j, size, size)

    def __getitem__(self, idx: int):
        # 1. Load Image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = self._get_scale_factor(w, h)
        img = img.resize((w // scale, h // scale), Image.LANCZOS)

        # Random Crop -> Tensor
        # (복잡한 열화 및 증강은 GPU의 augment.py에서 수행)
        img_crop = self._get_random_crop(img, size=512)
        return TF.to_tensor(img_crop)
