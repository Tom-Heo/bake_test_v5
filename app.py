from __future__ import annotations

import os
import tempfile
import zipfile
import torch
import numpy as np
import imageio.v3 as iio
import gradio as gr

# Local Modules
from config import Config
from core.net import BakeNet
from core.palette import Palette
from inference import (
    auto_detect_bit_depth,
    pad_image,
    unpad_image,
    save_tensor_to_16bit_png,
)


# =============================================================================
# [1] Model Loading
# =============================================================================
_cache: dict = {}


def _load_model():
    """BakeNet 로드. 체크포인트 파일이 갱신되면 자동 재로드."""
    ckpt_path = Config.LAST_CKPT_PATH

    if not os.path.exists(ckpt_path):
        return None, None, None, None

    mtime = os.path.getmtime(ckpt_path)
    if _cache.get("mtime") == mtime:
        return (
            _cache["model"],
            _cache["to_oklabp"],
            _cache["to_rgb"],
            _cache["device"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BakeNet(dim=Config.INTERNAL_DIM).to(device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if "ema_shadow" in ckpt:
            model.load_state_dict(ckpt["ema_shadow"], strict=False)
        else:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

        model.eval()
        to_oklabp = Palette.sRGBtoOklabP().to(device)
        to_rgb = Palette.OklabPtosRGB().to(device)

        _cache.update(
            mtime=mtime,
            model=model,
            to_oklabp=to_oklabp,
            to_rgb=to_rgb,
            device=device,
        )
        return model, to_oklabp, to_rgb, device

    except Exception as e:
        print(f"[Bake] Checkpoint load failed: {e}")
        return None, None, None, device


def _model_status() -> str:
    """현재 모델 상태 문자열."""
    model, _, _, device = _load_model()
    if model is not None:
        return f"BakeNet Loaded | {device}"
    return "Model Not Found -- checkpoints/last.pth 필요"


# =============================================================================
# [2] Image I/O
# =============================================================================
def _read_image(path: str) -> np.ndarray:
    """이미지 파일 -> numpy (H,W,3). 알파 채널 자동 제거."""
    ext = os.path.splitext(path)[1]
    try:
        img = iio.imread(path, extension=ext)
    except Exception:
        img = iio.imread(path)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _normalize(img_np: np.ndarray, bit_depth: str) -> tuple[torch.Tensor, str]:
    """numpy -> [0,1] 텐서 + 비트 심도 메시지."""
    img_f = img_np.astype(np.float32)

    if img_np.dtype == np.uint16:
        if bit_depth == "Auto":
            depth = auto_detect_bit_depth(img_np)
            msg = f"Auto {depth}-bit"
        else:
            depth = int(bit_depth)
            msg = f"Manual {depth}-bit"
        img_f /= (2**depth) - 1

    elif img_np.dtype == np.uint8:
        img_f /= 255.0
        msg = "8-bit"

    else:
        msg = f"{img_np.dtype}"

    tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
    return tensor.clamp(0.0, 1.0), msg


def _infer(tensor: torch.Tensor, model, to_oklabp, to_rgb, device) -> torch.Tensor:
    """BakeNet 추론. sRGB [0,1] -> 복원 sRGB [0,1]."""
    tensor = tensor.to(device)
    padded, org_size = pad_image(tensor)

    with torch.no_grad():
        oklabp = to_oklabp(padded)
        restored = model(oklabp)
        rgb = to_rgb(restored)

    return unpad_image(rgb, org_size).clamp(0.0, 1.0).cpu()


def _to_display(tensor: torch.Tensor) -> np.ndarray:
    """텐서 (1,3,H,W) [0,1] -> numpy (H,W,3) uint8."""
    return tensor.squeeze(0).permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().numpy()


# =============================================================================
# [3] Processing
# =============================================================================
def process_single(image_path: str | None, bit_depth: str):
    """단일 이미지 처리 -> (before, after, download_path, status)."""
    if image_path is None:
        raise gr.Error("이미지를 업로드해 주세요.")

    model, to_oklabp, to_rgb, device = _load_model()
    if model is None:
        raise gr.Error(
            "BakeNet을 로드할 수 없습니다. checkpoints/last.pth를 확인하세요."
        )

    try:
        img_np = _read_image(image_path)
        input_tensor, msg = _normalize(img_np, bit_depth)
        output_tensor = _infer(input_tensor, model, to_oklabp, to_rgb, device)
    except gr.Error:
        raise
    except Exception as e:
        torch.cuda.empty_cache()
        raise gr.Error(f"처리 실패: {e}")

    before = _to_display(input_tensor)
    after = _to_display(output_tensor)

    # 16-bit PNG 다운로드 파일 생성
    tmp = tempfile.NamedTemporaryFile(suffix="_bake_16bit.png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    save_tensor_to_16bit_png(output_tensor, tmp_path)

    return before, after, tmp_path, f"Done. ({msg})"


def process_batch(
    files: list[str] | None,
    bit_depth: str,
    progress=gr.Progress(),
):
    """배치 처리 -> (gallery, zip_path, status)."""
    if not files:
        raise gr.Error("파일을 업로드해 주세요.")

    model, to_oklabp, to_rgb, device = _load_model()
    if model is None:
        raise gr.Error(
            "BakeNet을 로드할 수 없습니다. checkpoints/last.pth를 확인하세요."
        )

    tmp_dir = tempfile.mkdtemp(prefix="bake_batch_")
    gallery: list[tuple[np.ndarray, str]] = []
    processed = 0

    for file_path in progress.tqdm(files, desc="Processing"):
        name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            img_np = _read_image(file_path)
            tensor, _ = _normalize(img_np, bit_depth)
            result = _infer(tensor, model, to_oklabp, to_rgb, device)

            save_tensor_to_16bit_png(result, os.path.join(tmp_dir, f"{name}_bake.png"))
            gallery.append((_to_display(result), name))
            processed += 1
        except Exception as e:
            print(f"[Bake] Skip {name}: {e}")
            torch.cuda.empty_cache()

    # ZIP 생성
    zip_path = os.path.join(tmp_dir, "bake_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(tmp_dir):
            if fname.endswith("_bake.png"):
                zf.write(os.path.join(tmp_dir, fname), fname)

    return gallery, zip_path, f"Done. {processed}/{len(files)} processed."


# =============================================================================
# [4] Theme
# =============================================================================
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#FEF2F2",
        c100="#FEE2E2",
        c200="#FECACA",
        c300="#FCA5A5",
        c400="#F87171",
        c500="#EF4444",
        c600="#D41201",
        c700="#B91C1C",
        c800="#991B1B",
        c900="#7F1D1D",
        c950="#450A0A",
    ),
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


# =============================================================================
# [5] UI
# =============================================================================
with gr.Blocks(theme=theme, title="Bake") as demo:

    gr.Markdown("# Bake\n*Accurate, therefore beautiful.*")

    with gr.Tabs():
        # ---- Single Image ----
        with gr.TabItem("Single Image"):
            with gr.Row():
                with gr.Column(scale=3):
                    single_input = gr.Image(
                        type="filepath",
                        label="Upload Image",
                        sources=["upload"],
                    )
                with gr.Column(scale=1):
                    single_bit = gr.Dropdown(
                        choices=["Auto", "8", "10", "12", "14", "16"],
                        value="Auto",
                        label="Bit Depth",
                        info="Auto: 픽셀 강도 기반 자동 감지. 어두운 10/12-bit 소스는 수동 지정 권장.",
                    )
                    single_btn = gr.Button("Process", variant="primary")
                    single_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                single_before = gr.Image(label="Before", interactive=False)
                single_after = gr.Image(label="After", interactive=False)

            single_download = gr.File(label="16-bit PNG Download", interactive=False)

            single_btn.click(
                fn=process_single,
                inputs=[single_input, single_bit],
                outputs=[
                    single_before,
                    single_after,
                    single_download,
                    single_status,
                ],
            )

        # ---- Batch Processing ----
        with gr.TabItem("Batch Processing"):
            with gr.Row():
                with gr.Column(scale=3):
                    batch_input = gr.File(
                        file_count="multiple",
                        label="Upload Images (PNG, TIFF, JPG, DPX)",
                        file_types=[
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".tif",
                            ".tiff",
                            ".dpx",
                        ],
                    )
                with gr.Column(scale=1):
                    batch_bit = gr.Dropdown(
                        choices=["Auto", "8", "10", "12", "14", "16"],
                        value="Auto",
                        label="Bit Depth",
                        info="시퀀스 작업 시 수동 지정으로 플리커 방지.",
                    )
                    batch_btn = gr.Button("Process All", variant="primary")
                    batch_status = gr.Textbox(label="Status", interactive=False)

            batch_gallery = gr.Gallery(
                label="Results",
                columns=4,
                object_fit="contain",
                height="auto",
            )
            batch_download = gr.File(label="Download All (ZIP)", interactive=False)

            batch_btn.click(
                fn=process_batch,
                inputs=[batch_input, batch_bit],
                outputs=[batch_gallery, batch_download, batch_status],
            )

    gr.Markdown(f"---\nBake v5 | {_model_status()}")


# =============================================================================
# [6] Launch
# =============================================================================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
