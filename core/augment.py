import torch
import torch.nn as nn
import random

from .palette import Palette


class BakeAugment(nn.Module):
    """
    [Bake GPU Darkroom]
    GPU-Accelerated Degradation Pipeline
    """

    def __init__(self):
        super().__init__()

        # Color Space Converter (sRGB -> OklabP)
        self.to_oklabp = Palette.sRGBtoOklabP()

    # -----------------------------------------------------------------
    # OklabP Random Curve Degradation
    # -----------------------------------------------------------------
    def _make_random_curve(self, n_ctrl=199, strength=0.10, device="cpu"):
        """
        Random Monotonic Piecewise-Linear Curve [-1, 1] -> [-1, 1].
        Returns (ctrl_x, ctrl_y) each of shape (n_ctrl + 2,).
        Endpoints are fixed at (-1, -1) and (1, 1).
        """
        ctrl_x = torch.linspace(-1.0, 1.0, n_ctrl + 2, device=device)
        ctrl_y = ctrl_x.clone()
        ctrl_y[1:-1] += torch.randn(n_ctrl, device=device) * strength
        ctrl_y, _ = torch.sort(ctrl_y)
        ctrl_y = ctrl_y.clamp(-1.0, 1.0)
        ctrl_y[0] = -1.0
        ctrl_y[-1] = 1.0
        return ctrl_x, ctrl_y

    def _apply_curve(self, values, ctrl_x, ctrl_y):
        """
        Apply piecewise-linear curve via searchsorted + lerp.
        values: arbitrary shape tensor
        ctrl_x, ctrl_y: (K,) sorted control points
        """
        shape = values.shape
        flat = values.reshape(-1).clamp(ctrl_x[0], ctrl_x[-1])
        idx = torch.searchsorted(ctrl_x, flat, right=True) - 1
        idx = idx.clamp(0, len(ctrl_x) - 2)
        x0, x1 = ctrl_x[idx], ctrl_x[idx + 1]
        y0, y1 = ctrl_y[idx], ctrl_y[idx + 1]
        t = (flat - x0) / (x1 - x0 + 1e-8)
        return (y0 + t * (y1 - y0)).reshape(shape)

    def apply_oklabp_curve(self, oklabp):
        """
        Random Per-Channel Curve Distortion in OklabP Space.
        Input / Output: (B, 3, H, W) OklabP [-1, 1]
        """
        strengths = [0.25, 0.15, 0.15]  # Lp / ap / bp
        for ch in range(3):
            ctrl_x, ctrl_y = self._make_random_curve(
                n_ctrl=5, strength=strengths[ch], device=oklabp.device
            )
            oklabp[:, ch] = self._apply_curve(oklabp[:, ch], ctrl_x, ctrl_y)
        return oklabp

    def forward(self, x):
        """
        Input:  (B, 3, H, W) sRGB [0, 1]
        Returns: (Input_OklabP, Target_OklabP)  both [-1, 1]
        """
        # --- [Geometric Augmentation] ---
        # Flip & Rotate (GPU)
        if random.random() < 0.5:
            x = torch.flip(x, [3])  # H-Flip
        if random.random() < 0.5:
            x = torch.flip(x, [2])  # V-Flip
        if random.random() < 0.5:
            x = torch.rot90(x, 1, [2, 3])

        # sRGB -> OklabP (single conversion, no round-trip)
        target = self.to_oklabp(x)
        input_t = target.clone()

        # --- [Degradation Pipeline] ---

        # OklabP Curve Distortion (Input-only)
        input_t = self.apply_oklabp_curve(input_t)

        return input_t, target
