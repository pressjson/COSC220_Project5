#!/usr/bin/env python3

from pathlib import Path

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

# Model IDs. Uncomment the one you want to use.
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
# MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
# MODEL_ID = "black-forest-labs/FLUX.2-klein-9B-KV"
# MODEL_ID = "black-forest-labs/FLUX.2-dev"


def load_pipe() -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()  # helps when VRAM is tight
    return pipe


def resize_for_flux(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """
    Resize while preserving aspect ratio.
    Snap dimensions to multiples of 32 to avoid awkward shapes.
    """
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(32, round((w * scale) / 32) * 32)
    new_h = max(32, round((h * scale) / 32) * 32)
    return img.resize((new_w, new_h), Image.LANCZOS)


def flux2_edit_image(
    pipe: Flux2KleinPipeline,
    input_img: Image.Image,
    prompt: str = (
        "Colorize this grayscale image realistically. "
        "Preserve the exact composition, pose, framing, and object layout. "
        "Do not add or remove objects. Use plausible natural colors."
    ),
    seed: int = 0,
) -> Image.Image:
    img = resize_for_flux(input_img, max_side=1024)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        image=img,
        prompt=prompt,
        width=img.width,
        height=img.height,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]

    return result


if __name__ == "__main__":
    pipe = load_pipe()

    flux2_edit_image(
        pipe,
        input_path="images/IMG_0956_sklearn_grayscale.JPEG",
        seed=42,
    )
