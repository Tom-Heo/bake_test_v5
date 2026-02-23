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
        self, B, n_ctrl=399, strength=0.10, device="cpu", dtype=torch.float32
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
        self, B, n_ctrl=33, strength=0.10, device="cpu", dtype=torch.float32
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
    # 2. HSL 2D Grid Operations (Polar Coordinate Distortion)
    # =================================================================
    def _make_hsl_grid(self, B, strength, ctrl_res, device, dtype):
        """
        [극좌표계 색상 왜곡 맵 (Polar Color-Space LUT)]
        2D 색공간(a, b 평면) 전체를 위상학적으로 부드럽게 뒤틀기 위한 제어 변수를 생성합니다.
        무채색의 절대성을 보존하고 채도의 비율적 깊이를 유지하기 위해,
        직교 좌표계의 가산(+) 왜곡이 아닌 극좌표계의 스케일(Scale)과 회전(Rotation) 물리량을 정의합니다.
        """
        # 1. 극소수의 제어점(3x3 ~ 5x5)에서 색공간을 통째로 밀어낼 거시적인 파동의 씨앗을 생성합니다.
        W_lum_low = torch.randn(
            B, 1, ctrl_res, ctrl_res, device=device, dtype=dtype
        ) * (strength * 0.5)
        W_scale_low = (
            torch.randn(B, 1, ctrl_res, ctrl_res, device=device, dtype=dtype) * strength
        )
        W_hue_low = (
            torch.randn(B, 1, ctrl_res, ctrl_res, device=device, dtype=dtype) * strength
        )

        # 2. Bicubic 보간을 통해 매끄럽고 연속적인 2D 색상 변환 맵을 완성합니다.
        W_lum = F.interpolate(
            W_lum_low, size=(self.G, self.G), mode="bicubic", align_corners=True
        ).squeeze(1)
        W_scale = F.interpolate(
            W_scale_low, size=(self.G, self.G), mode="bicubic", align_corners=True
        ).squeeze(1)
        W_hue = F.interpolate(
            W_hue_low, size=(self.G, self.G), mode="bicubic", align_corners=True
        ).squeeze(1)

        # 3. 물리적 불변성을 확보하기 위해 극좌표계의 척도로 변환합니다.
        # 채도는 1.0을 기점으로 곱해지는 승수(Multiplier)로, 색조는 라디안(Radian) 단위의 회전각이 됩니다.
        delta_L = W_lum
        scale_C = 1.0 + W_scale
        rot_h = W_hue * torch.pi

        return torch.stack([delta_L, scale_C, rot_h], dim=1)

    def apply_hsl(self, input_t, target_t, strength=0.10):
        """
        [원본 조건부 극좌표계 HSL 왜곡 (Conditional Polar HSL Shift)]
        실제 사진 보정에서 발생하는 '특정 색상 영역의 틀어짐'을 모사합니다.
        왜곡의 기준점(Condition)을 항상 '원본(Target)'에 두어,
        AI가 픽셀 간의 구조적 관계를 통해 명확한 역함수를 추론할 수 있도록 유도합니다.
        """
        B = input_t.shape[0]
        device, dtype = input_t.device, input_t.dtype

        # 1. 색공간을 분할하는 주파수를 동적으로 할당하여(3~5),
        # 광범위한 톤 변화부터 국소적인 색상 틀어짐까지 다채로운 패턴을 생성합니다.
        ctrl_res = random.randint(3, 5)
        polar_grid = self._make_hsl_grid(B, strength, ctrl_res, device, dtype)

        # 2. 원본(Target)의 깨끗한 a, b 좌표를 나침반 삼아,
        # 2D 왜곡 맵에서 해당 픽셀이 받아야 할 극좌표 변형의 크기를 샘플링합니다.
        ab_coords_tgt = target_t[:, 1:3, :, :].permute(0, 2, 3, 1)

        sampled_vars = F.grid_sample(
            polar_grid,
            ab_coords_tgt,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        delta_L = sampled_vars[:, 0, :, :]
        scale_C = sampled_vars[:, 1, :, :]
        rot_h = sampled_vars[:, 2, :, :]

        # 3. 망가진 입력(Input)의 직교 좌표를 극좌표(채도, 색조)로 치환합니다.
        L_in, a_in, b_in = torch.unbind(input_t, dim=1)

        # 원점(무채색)에서의 미분 불가능성과 0으로 나누기 오류를 방지합니다.
        C_in = torch.sqrt(a_in**2 + b_in**2 + 1e-8)
        h_in = torch.atan2(b_in, a_in)

        # 4. 물리적 왜곡 적용
        # 채도의 붕괴 없이 본연의 깊이를 조율하고, 색조의 위상을 부드럽게 회전시킵니다.
        L_out = L_in + delta_L
        C_out = C_in * scale_C
        h_out = h_in + rot_h

        # 5. 직교 좌표계 복원
        # 정보의 비가역적 소실을 막기 위해 클리핑(Clamp)을 배제하여 무한한 그라디언트를 보존합니다.
        a_out = C_out * torch.cos(h_out)
        b_out = C_out * torch.sin(h_out)

        return torch.stack([L_out, a_out, b_out], dim=1)

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
    def apply_oklabp_curve(self, input_t, target_t, strength=0.10):
        """
        [원본 조건부 글로벌 톤 & 화이트 밸런스 왜곡]
        L(명도) 채널: 원본을 기준으로 1D S/역S 곡선을 적용하여 비선형적으로 왜곡합니다.
        a, b(색상) 채널: 강도(strength)에 비례하는 단일 스칼라 오프셋(Translation)을 더해
        전역적인 색온도 및 틴트의 틀어짐을 모사합니다.
        """
        B = input_t.shape[0]
        device, dtype = input_t.device, input_t.dtype

        L_tgt = target_t[:, 0, :, :]
        L_in, a_in, b_in = torch.unbind(input_t, dim=1)

        # 1. L 채널: 명암 및 대비의 비선형 왜곡 (Global Tone Curve)
        # 통제 변수 strength를 곡선의 최대 진폭 한계치로 전달합니다.
        ctrl_x_L, ctrl_y_L = self._make_random_curve(B, 399, strength, device, dtype)
        delta_L = self._apply_curve(L_tgt, ctrl_x_L, ctrl_y_L) - L_tgt
        L_out = L_in + delta_L

        # 2. a, b 채널: 전역 화이트 밸런스 틀어짐 (Global Color Shift)
        # strength를 기준으로 평행 이동의 범위를 [-strength/2, +strength/2]로 동기화합니다.
        shift_a = (torch.rand(B, 1, 1, device=device, dtype=dtype) * strength) - (
            strength / 2.0
        )
        shift_b = (torch.rand(B, 1, 1, device=device, dtype=dtype) * strength) - (
            strength / 2.0
        )

        a_out = a_in + shift_a
        b_out = b_in + shift_b

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
            lambda inp: self.apply_oklabp_curve(inp, target, strength=0.10),
            lambda inp: self.apply_hsl(inp, target, strength=0.10),
            lambda inp: self.apply_color_wheels(inp, target, strength=0.10),
        ]

        random.shuffle(degradations)

        for apply_degradation in degradations:
            input_t = apply_degradation(input_t)

        return input_t, target
