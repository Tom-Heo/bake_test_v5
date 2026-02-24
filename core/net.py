from __future__ import annotations

import torch
import torch.nn as nn
from .heo import Heo
from .bake import (
    OklabPtoBakedBaseColor,
    BakedBaseColortoColorEmbedding,
    BakedBaseColortoPerceptualOklabP,
)


class BakeNet(nn.Module):
    def __init__(self, depth=30, dim=30):
        super().__init__()
        self.depth = depth
        self.dim = dim

        # 1. Stem (The Anchor)
        # 3ch -> 30ch Internal Base
        self.stem = OklabPtoBakedBaseColor()

        # 2. Components for the Body (The Loop)

        # A. Embedding Module
        # 원본(Base)에서 문맥을 추출
        self.embedding_module = BakedBaseColortoColorEmbedding(dim)

        # B. Context Gate
        # 추출된 문맥(Embedding)과 주 흐름(Stream)을 혼합
        self.context_gate = Heo.HeoGate2d(dim)

        # C. NeMO Blocks
        # 국소적 디테일 복원 — 대칭 커널 패턴
        _nemo_cycle = [
            Heo.NeMO33,
            Heo.NeMO55,
            Heo.NeMO77,
            Heo.NeMO99,
            Heo.NeMO1111,
            Heo.NeMO1111,
            Heo.NeMO99,
            Heo.NeMO77,
            Heo.NeMO55,
            Heo.NeMO33,
        ]
        self.nemo_modules = nn.ModuleList(
            [cls(dim) for cls in _nemo_cycle for _ in range(depth // 10)]
        )

        # D. Residual Gates
        # 연산 전후의 차이를 학습 (Gradient Flow 보장)
        self.residual_gates = nn.ModuleList([Heo.HeoGate2d(dim) for _ in range(depth)])

        # 3. Head (The Finish)
        # 30ch -> 3ch Output
        self.head = BakedBaseColortoPerceptualOklabP(in_dim=dim, out_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3, H, W) - OklabP
        Output: (B, 3, H, W) - Refined OklabP
        """
        # A. Create the Base (Immutable Reference)
        base = self.stem(x)

        # B. Initialization (0-th Block)
        # 1. Embedding
        emb0 = self.embedding_module(base)

        # 2. Gating: [Embedding vs Base]
        # 지시사항: "context_gate(emb0, base)"
        # 임베딩(emb0)을 주체로 하되, 원본(base)을 참조하여 초기 신호를 형성
        feat = self.context_gate(emb0, base)

        # 3. NeMO Processing
        out0 = self.nemo_modules[0](feat)

        # 4. Residual Connection
        # NeMO 출력과 입력(feat)을 결합하여 흐름을 시작
        feat = self.residual_gates[0](out0, feat)

        # C. The Loop
        for i in range(1, self.depth):

            # 1. NeMO Processing
            nemo_out = self.nemo_modules[i](feat)

            # 2. Residual Connection
            feat = self.residual_gates[i](nemo_out, feat)

        # D. Residual + Head (0.25x Boost)
        return x + 0.25 * self.head(feat)
