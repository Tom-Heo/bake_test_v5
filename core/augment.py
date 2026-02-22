import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .palette import Palette


class BakeAugment(nn.Module):
    """
    [Bake GPU Darkroom]
    AI 색상 복원 모델 학습을 위한 GPU 가속 기반 사진 열화(Degradation) 파이프라인.

    인위적인 픽셀 파괴(White Noise, Blur)를 배제하고, 실제 카메라 센서의 한계와
    잘못된 보정(Curve, HSL, Color Wheels)에서 발생하는 '위상학적으로 연속적인 색상 왜곡'을 생성합니다.
    특히, 열화의 기준점(Condition)을 항상 '원본 이미지(Target)'에 두어,
    AI가 피사체의 구조(Structure)를 단서로 삼아 명확한 역함수를 학습할 수 있도록 설계되었습니다.
    """

    def __init__(self, hsl_grid_size=33):
        super().__init__()
        # 시각적으로 균일한(Perceptually Uniform) OklabP 공간을 활용하여
        # 인간의 인지와 수학적 연산의 궤를 완벽히 일치시킵니다. (값 범위: [-1, 1])
        self.to_oklabp = Palette.sRGBtoOklabP()
        self.G = hsl_grid_size

    # =================================================================
    # 1. 1D Curve Generators & Application (Global Tone & Color)
    # =================================================================
    def _make_random_curve(
        self, B, n_ctrl=399, strength=1.00, device="cpu", dtype=torch.float32
    ):
        """
        [글로벌 톤 커브 (Global Tone Curve) - 동적 진폭 제어 적용]
        단조 증가(Monotonic)를 보장하는 매끄러운 S자/역S자 곡선을 생성합니다.
        배치 내 이미지마다 대각선(항등 함수)으로부터 이탈하는 강도를 무작위로 설정하여,
        완벽히 깨끗한 상태부터 극단적으로 망가진 다이나믹 레인지까지 연속적인 스펙트럼을 모사합니다.
        """
        ctrl_x = (
            torch.linspace(-1.0, 1.0, n_ctrl + 2, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )

        # 브라운 운동을 통한 매끄러운 파동 생성 및 정규화
        noise = torch.randn(B, n_ctrl + 1, device=device, dtype=dtype)
        brownian = torch.cumsum(noise, dim=1)
        brownian = brownian - brownian.mean(dim=1, keepdim=True)
        brownian = brownian / (brownian.std(dim=1, keepdim=True) + 1e-8)

        # [동적 진폭 제어]
        # 곡선의 휘어짐 정도를 결정합니다.
        # dynamic_strength가 0이 되면 brownian은 모두 0이 되고, exp(0)=1이 되어 완벽한 직선을 그립니다.
        dynamic_strength = torch.rand(B, 1, device=device, dtype=dtype) * strength
        brownian = brownian * dynamic_strength

        # 지수 함수를 통한 단조 증가 강제 및 누적
        steps = torch.exp(brownian)
        y_inner = torch.cumsum(steps, dim=1)
        y_full = torch.cat(
            [torch.zeros(B, 1, device=device, dtype=dtype), y_inner], dim=1
        )

        y_max = y_full[:, -1:]

        # [다이나믹 레인지 손실 동기화]
        # 곡선이 선형(원형)에 가까울 때는 블랙/화이트 오프셋도 발생하지 않도록,
        # 동적 진폭의 비율(Ratio)에 맞춰 오프셋의 한계치도 함께 축소시킵니다.
        offset_ratio = dynamic_strength / (strength + 1e-8)
        black_offset = (
            torch.rand(B, 1, device=device, dtype=dtype) * 0.05 * offset_ratio
        )
        white_offset = (
            torch.rand(B, 1, device=device, dtype=dtype) * 0.05 * offset_ratio
        )

        scale = 2.0 - (black_offset + white_offset)
        ctrl_y = (y_full / y_max) * scale - 1.0 + black_offset

        return ctrl_x, ctrl_y

    def _make_random_walk(
        self, B, n_ctrl=33, strength=1.00, device="cpu", dtype=torch.float32
    ):
        """
        [스플릿 토닝 커브 (Color Wheels Curve)]
        특정 명도 구간에 독립적인 색을 덧입히기 위한 자유 파동(Non-monotonic) 곡선을 생성합니다.
        배치 내 이미지마다 각기 다른 한계 진폭을 갖도록 설계하여 열화 강도의 다양성을 극대화합니다.
        """
        ctrl_x = (
            torch.linspace(-1.0, 1.0, n_ctrl + 2, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )

        noise = torch.randn(B, n_ctrl + 2, device=device, dtype=dtype)
        walk = torch.cumsum(noise, dim=1)
        walk = walk - walk.mean(dim=1, keepdim=True)

        # [동적 진폭 제어]
        # 최대 진폭을 0.0 ~ strength 사이의 무작위 값으로 설정하여 다채로운 오염 강도 부여
        dynamic_strength = torch.rand(B, 1, device=device, dtype=dtype) * strength
        max_val = walk.abs().max(dim=1, keepdim=True)[0] + 1e-8
        ctrl_y = (walk / max_val) * dynamic_strength

        return ctrl_x, ctrl_y

    def _apply_curve(self, values, ctrl_x, ctrl_y):
        """
        [GPU 최적화 1D 보간]
        탐색 알고리즘(searchsorted) 없이 균등 간격의 수학적 특성을 활용해
        O(1) 메모리 인덱싱만으로 곡선을 텐서에 맵핑합니다.
        """
        B, H, W = values.shape
        flat = values.view(B, -1)
        K = ctrl_x.shape[1]

        idx_float = ((flat + 1.0) / 2.0) * (K - 1)
        idx = idx_float.long().clamp(0, K - 2)

        x0 = torch.gather(ctrl_x, 1, idx)
        x1 = torch.gather(ctrl_x, 1, idx + 1)
        y0 = torch.gather(ctrl_y, 1, idx)
        y1 = torch.gather(ctrl_y, 1, idx + 1)

        t = (flat - x0) / (x1 - x0 + 1e-8)
        out = y0 + t * (y1 - y0)

        return out.view(B, H, W)

    # =================================================================
    # 2. HSL 2D Grid Operations (Local Color Distortion)
    # =================================================================
    def _make_hsl_grid(self, B, strength, ctrl_res, device, dtype):
        """
        [HSL 국소 색상 왜곡 벡터 필드]
        극소수의 제어점에서 생성된 노이즈를 방사형(채도) 및 접선형(색조) 벡터로 변환하여,
        특정 색상의 영역만 위상학적으로 부드럽게 뒤틀어버리는 2D 그리드를 완성합니다.
        """
        W_sat_low = (
            torch.randn(B, 1, ctrl_res, ctrl_res, device=device, dtype=dtype) * strength
        )
        W_hue_low = (
            torch.randn(B, 1, ctrl_res, ctrl_res, device=device, dtype=dtype) * strength
        )
        W_lum_low = torch.randn(
            B, 1, ctrl_res, ctrl_res, device=device, dtype=dtype
        ) * (strength * 0.5)

        W_sat = F.interpolate(
            W_sat_low, size=(self.G, self.G), mode="bicubic", align_corners=True
        ).squeeze(1)
        W_hue = F.interpolate(
            W_hue_low, size=(self.G, self.G), mode="bicubic", align_corners=True
        ).squeeze(1)
        W_lum = F.interpolate(
            W_lum_low, size=(self.G, self.G), mode="bicubic", align_corners=True
        ).squeeze(1)

        a_coords = torch.linspace(-1.0, 1.0, self.G, device=device, dtype=dtype)
        b_coords = torch.linspace(-1.0, 1.0, self.G, device=device, dtype=dtype)

        grid_a, grid_b = torch.meshgrid(a_coords, b_coords, indexing="xy")
        grid_a = grid_a.unsqueeze(0).expand(B, -1, -1)
        grid_b = grid_b.unsqueeze(0).expand(B, -1, -1)

        R_a, R_b = grid_a, grid_b
        T_a, T_b = -grid_b, grid_a

        delta_L = W_lum
        delta_a = (W_sat * R_a) + (W_hue * T_a)
        delta_b = (W_sat * R_b) + (W_hue * T_b)

        return torch.stack([delta_L, delta_a, delta_b], dim=1)

    def apply_hsl(self, input_t, target_t, strength=1.00):
        """
        [원본 조건부 HSL 적용]
        원본(Target)의 깨끗한 색상 좌표를 기준으로 오프셋(Delta)을 샘플링한 뒤,
        그 변화량만 망가진 이미지(Input)에 누적합니다.
        색상 덩어리(Frequency) 크기를 3, 4, 5 중 무작위로 결정하여 컬러 밴딩 방지와 패턴 다양성을 확보합니다.
        """
        B = input_t.shape[0]
        device, dtype = input_t.device, input_t.dtype

        # 색공간을 쪼개는 주파수 동적 할당 (3: 광범위한 변형, 5: 국소적인 변형)
        ctrl_res = random.randint(3, 5)

        offset_grid = self._make_hsl_grid(B, strength, ctrl_res, device, dtype)

        # 오프셋 샘플링의 기준점은 항상 '원본(Target)'의 a, b 좌표를 활용합니다.
        ab_coords_tgt = target_t[:, 1:3, :, :].permute(0, 2, 3, 1)

        delta_lab = F.grid_sample(
            offset_grid,
            ab_coords_tgt,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # 원본 구조 기반으로 계산된 델타를 망가진 이미지에 덧입힙니다 (Out-of-place)
        return input_t + delta_lab

    # =================================================================
    # 3. Global Color Wheels Operation (Split Toning)
    # =================================================================
    def apply_color_wheels(self, input_t, target_t, strength=1.00):
        """
        [원본 조건부 스플릿 토닝]
        원본 이미지의 명도(L) 대역을 평가하여 어두운 곳과 밝은 곳에 서로 다른 색 틴트를 씌웁니다.
        이미 뭉개진 명도가 아닌 '원본의 명도'를 기준으로 삼으므로, AI가 피사체의 윤곽을 복원 단서로 삼을 수 있습니다.
        """
        B = input_t.shape[0]
        device, dtype = input_t.device, input_t.dtype

        L_tgt = target_t[:, 0, :, :]
        L_in, a_in, b_in = torch.unbind(input_t, dim=1)

        # 원본 L값을 기준으로 a채널(Green-Red) 틴트 변화량 계산
        ctrl_L_a, ctrl_offset_a = self._make_random_walk(
            B, self.G, strength, device, dtype
        )
        delta_a = self._apply_curve(L_tgt, ctrl_L_a, ctrl_offset_a)

        # 원본 L값을 기준으로 b채널(Blue-Yellow) 틴트 변화량 계산
        ctrl_L_b, ctrl_offset_b = self._make_random_walk(
            B, self.G, strength, device, dtype
        )
        delta_b = self._apply_curve(L_tgt, ctrl_L_b, ctrl_offset_b)

        # 계산된 틴트 오프셋을 망가진 입력의 색상 채널에 누적
        a_out = a_in + delta_a
        b_out = b_in + delta_b

        return torch.stack([L_in, a_out, b_out], dim=1)

    # =================================================================
    # 4. Pipeline Execution
    # =================================================================
    def apply_oklabp_curve(self, input_t, target_t):
        """
        [원본 조건부 베이스 커브 적용]
        전체적인 대비와 전역 색온도 왜곡을 수행합니다.
        원본을 곡선에 통과시켜 이상적인 변화량(Delta)을 구한 뒤 적용하여,
        과도한 정보 소실로 인한 AI의 블러(Blur) 생성을 원천 차단합니다.
        """
        B = input_t.shape[0]
        device, dtype = input_t.device, input_t.dtype

        L_tgt, a_tgt, b_tgt = torch.unbind(target_t, dim=1)
        L_in, a_in, b_in = torch.unbind(input_t, dim=1)

        # 1. 명도(L) 대비 왜곡
        ctrl_x_L, ctrl_y_L = self._make_random_curve(B, 399, 1.00, device, dtype)
        delta_L = self._apply_curve(L_tgt, ctrl_x_L, ctrl_y_L) - L_tgt
        L_out = L_in + delta_L

        # 2. a채널(Green-Red) 균형 왜곡
        ctrl_x_a, ctrl_y_a = self._make_random_curve(B, 399, 1.00, device, dtype)
        delta_a = self._apply_curve(a_tgt, ctrl_x_a, ctrl_y_a) - a_tgt
        a_out = a_in + delta_a

        # 3. b채널(Blue-Yellow) 균형 왜곡
        ctrl_x_b, ctrl_y_b = self._make_random_curve(B, 399, 1.00, device, dtype)
        delta_b = self._apply_curve(b_tgt, ctrl_x_b, ctrl_y_b) - b_tgt
        b_out = b_in + delta_b

        return torch.stack([L_out, a_out, b_out], dim=1)

    def forward(self, x):
        """
        [Bake Augmentation 순전파]
        Input:  (B, 3, H, W) 포맷의 sRGB 텐서
        Returns: 망가진 이미지(Input)와 원본 이미지(Target)의 OklabP [-1, 1] 쌍
        """
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # --- [기하학적 증강 (Geometric Augmentation)] ---
        # CPU-GPU 동기화 지연을 방지하기 위해 Python 제어문을 배제하고
        # 순수 GPU 텐서 마스킹(Tensor Masking) 방식의 병렬 플립(Flip) 연산을 수행합니다.
        flip_h_mask = torch.rand(B, 1, 1, 1, device=device, dtype=dtype) < 0.5
        x = torch.where(flip_h_mask, torch.flip(x, [3]), x)

        flip_v_mask = torch.rand(B, 1, 1, 1, device=device, dtype=dtype) < 0.5
        x = torch.where(flip_v_mask, torch.flip(x, [2]), x)

        # --- [색공간 변환 (Color Space Conversion)] ---
        target = self.to_oklabp(x)
        input_t = target.clone()

        # --- [순차적 열화 파이프라인 (Degradation Pipeline)] ---
        degradations = [
            lambda inp: self.apply_oklabp_curve(inp, target),
            lambda inp: self.apply_hsl(inp, target, strength=1.00),
            lambda inp: self.apply_color_wheels(inp, target, strength=1.00),
        ]

        random.shuffle(degradations)

        for apply_degradation in degradations:
            input_t = apply_degradation(input_t)

        input_t = input_t.clamp(-1.0, 1.0)

        return input_t, target
