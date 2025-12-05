from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from models import MODEL_REGISTRY


# ---------- Streamlit helpers -------------------------------------------------------------------

st.set_page_config(page_title="Photo Restoration Studio", page_icon="🎨", layout="wide")


def available_devices() -> Tuple[str, ...]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    if torch.backends.mps.is_available():
        # keep GPU options at the front
        if "mps" not in devices:
            devices.insert(0, "mps")
    # deduplicate while keeping order
    seen = []
    for dev in devices:
        if dev not in seen:
            seen.append(dev)
    return tuple(seen)


def normalize_state_dict(raw_state):
    if isinstance(raw_state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in raw_state:
                raw_state = raw_state[key]
                break

    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint is not a valid state_dict")

    clean_state = {}
    for k, v in raw_state.items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        clean_state[new_key] = v
    return clean_state


@st.cache_resource(show_spinner=False)
def load_model(task_key: str, device: str):
    cfg = MODEL_REGISTRY[task_key]
    model = cfg["constructor"](**cfg.get("init_kwargs", {}))
    checkpoint = torch.load(cfg["checkpoint"], map_location=device)
    state_dict = normalize_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------- Image helpers ----------------------------------------------------------------------

def resize_with_limit(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    largest = max(w, h)
    if largest <= max_side:
        return image
    scale = max_side / largest
    new_size = (max(4, int(w * scale)), max(4, int(h * scale)))
    return image.resize(new_size, Image.LANCZOS)


def ensure_divisible(image: Image.Image, divisor: int) -> Image.Image:
    if divisor <= 1:
        return image
    w, h = image.size
    new_w = max(divisor, (w // divisor) * divisor)
    new_h = max(divisor, (h // divisor) * divisor)
    if new_w == w and new_h == h:
        return image
    return image.resize((new_w, new_h), Image.LANCZOS)


def pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    arr = (arr * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_base_image(image: Image.Image, max_side: int, divisor: int) -> Image.Image:
    processed = resize_with_limit(image.convert("RGB"), max_side)
    processed = ensure_divisible(processed, divisor)
    return processed


def prepare_mask(mask_img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    mask = mask_img.convert("L").resize(size, Image.NEAREST)
    mask_np = np.array(mask).astype(np.float32) / 255.0
    # training used 1 for valid pixels, 0 for holes
    mask_np = (mask_np > 0.5).astype(np.float32)
    return mask_np


# ---------- Task-specific pipelines ------------------------------------------------------------

def run_super_resolution(image: Image.Image, model, device: str, max_side: int, divisor: int):
    lr_image = prepare_base_image(image, max_side, divisor)
    lr_tensor = pil_to_tensor(lr_image, device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_image = tensor_to_pil(sr_tensor)
    return lr_image, sr_image


def run_denoise(image: Image.Image, model, device: str, max_side: int, divisor: int):
    noisy_image = prepare_base_image(image, max_side, divisor)
    noisy_tensor = pil_to_tensor(noisy_image, device)
    with torch.no_grad():
        clean_tensor = model(noisy_tensor)
    clean_image = tensor_to_pil(clean_tensor)
    return noisy_image, clean_image


def run_colorization(image: Image.Image, model, device: str, max_side: int, divisor: int):
    gray_image = prepare_base_image(image.convert("RGB"), max_side, divisor)
    lab = cv2.cvtColor(np.array(gray_image), cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    L_tensor = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        ab_pred = model(L_tensor.to(device))
    ab = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab = (ab * 128 + 128).clip(0, 255).astype(np.uint8)
    lab_out = np.concatenate([L[..., None], ab], axis=2)
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    colorized = Image.fromarray(rgb_out)
    return gray_image.convert("L"), colorized


def run_inpaint(
    image: Image.Image,
    mask_img: Image.Image,
    model,
    device: str,
    max_side: int,
    divisor: int,
):
    base_image = prepare_base_image(image, max_side, divisor)
    mask_np = prepare_mask(mask_img, base_image.size)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    base_np = np.array(base_image).astype(np.float32) / 255.0
    masked_np = base_np * mask_np[..., None]
    masked_tensor = torch.from_numpy(masked_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        restored_tensor = model(masked_tensor, mask_tensor)

    restored = tensor_to_pil(restored_tensor)
    masked_preview = Image.fromarray((masked_np * 255).astype(np.uint8))
    return masked_preview, restored


TASK_TO_RUNNER = {
    "super_resolution": run_super_resolution,
    "denoise": run_denoise,
    "colorize": run_colorization,
    "inpaint": run_inpaint,
}


# ---------- UI ---------------------------------------------------------------------------------

def main():
    st.title("🎨 Photo Restoration Studio")
    st.caption(
        "Upload a photo, pick one of your trained models, and preview the restored result side-by-side."
    )

    sidebar = st.sidebar
    sidebar.header("Configuration")

    device_choices = available_devices()
    task_labels = {cfg["display_name"]: key for key, cfg in MODEL_REGISTRY.items()}
    selected_label = sidebar.selectbox("Task", list(task_labels.keys()))
    task_key = task_labels[selected_label]
    device = sidebar.selectbox("Compute Device", device_choices, index=0)
    max_side = sidebar.slider("Max preview size (px)", 256, 1024, 640, step=32)
    preview_width = sidebar.slider("On-screen preview width (px)", 200, 600, 380, step=20)

    cfg = MODEL_REGISTRY[task_key]
    sidebar.info(cfg["description"])

    mask_file = None
    if task_key == "inpaint":
        mask_file = sidebar.file_uploader(
            "Mask image (white = keep, black = fill)", type=["png", "jpg", "jpeg"], key="mask"
        )
        sidebar.markdown(
            "Tip: Create a quick mask in any editor. White pixels keep the image; black pixels are replaced."
        )

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

    if uploaded is None:
        st.info("Choose an image to get started. Large photos will be resized using the slider above.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.write(f"File: **{uploaded.name}** — {image.size[0]}×{image.size[1]}")

    divisor = cfg.get("processing_divisor", 4)

    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        st.subheader("Input Preview")
        st.image(resize_with_limit(image, max_side), width=preview_width)

    result_slot = col2.empty()

    if task_key == "inpaint" and mask_file is None:
        st.warning("Upload a binary mask to run inpainting.")
        return

    if st.button("Run Model", type="primary"):
        with st.spinner("Running inference…"):
            model = load_model(task_key, device)
            runner = TASK_TO_RUNNER[task_key]
            if task_key == "inpaint":
                mask_image = Image.open(mask_file).convert("L")
                preview, output = runner(image, mask_image, model, device, max_side, divisor)
            else:
                preview, output = runner(image, model, device, max_side, divisor)

        with col1:
            st.image(preview, caption="Model input", width=preview_width)

        with col2:
            st.subheader("Result")
            result_slot.image(output, width=preview_width)
            st.download_button(
                "Download PNG",
                data=pil_to_bytes(output),
                file_name=f"restored_{task_key}_{uploaded.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
            )

        if task_key == "super_resolution":
            scale = cfg.get("init_kwargs", {}).get("scale", 2)
            st.success(f"Done! Output size: {output.size[0]}×{output.size[1]} (≈×{scale} upscale).")
        else:
            st.success("Done! Download the result or tweak the settings to iterate further.")

    with st.expander("Usage notes"):
        st.markdown(
            "- **Super Resolution / Denoise**: Works best on downsampled or noisy scans."
            "\n- **Colorization**: Upload grayscale photos (color shots will be desaturated automatically)."
            "\n- **Inpainting**: Provide a mask where the missing area is black. The app keeps everything else intact."
            "\n- Use the *Max preview size* slider to balance fidelity and inference speed."
        )


if __name__ == "__main__":
    main()
