#!/usr/bin/env python3

"""Convert images from color to grayscale."""

import os
import numpy as np
from PIL import Image
from skimage import color


def load_image(path: str) -> Image.Image:
    """Load an image."""
    image = Image.open(path)

    return image


def save_image(img: Image.Image, path: str) -> None:
    """Save an image."""
    img.save(path)


def pil_grayscale(img: Image.Image) -> Image.Image:
    """Use PIL.Image.convert("L") for grayscale conversion."""
    grayscale_img = img.convert("L")

    return grayscale_img


def skimage_grayscale(img: Image.Image) -> Image.Image:
    """Use skimage.color.rgb2lab() for grayscale conversion."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    l_channel = lab[..., 0]

    gray = np.clip((l_channel / 100.0) * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(gray, mode='L')


def convert_directory(input_dir: str, output_dir: str, mode=True) -> None:
    """
    Convert a directory to grayscale, excluding subdirs.

    If mode = True (default), use skimage_grayscale; else use pil_grayscale.
    """
    os.mkdir(output_dir, exist_ok=True)

    for entry in os.scandir(input_dir):
        if not entry.is_file():
            print(f"Warning: Not using subdirs {entry.name} idk why it's here")
            continue
        img = load_image(entry.path)
        if mode:
            grayscale_img = skimage_grayscale(img)
        else:
            grayscale_img = pil_grayscale(img)
        save_image(grayscale_img, os.path.join(output_dir, entry.path))


def convert_image(input_path: str, output_path: None | str = None, mode=True) -> Image.Image:
    """
    Convert a single image to grayscale.

    If mode = True (default), use skimage_grayscale; else use pil_grayscale.
    output_path is optional, does not write to disk if omitted.
    """
    img = load_image(input_path)
    gs_img = skimage_grayscale(img) if mode else pil_grayscale(img)
    if output_path:
        save_image(gs_img, output_path)
    return gs_img
