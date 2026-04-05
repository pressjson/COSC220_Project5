#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin

# Comes from the DDColor repo/package
from ddcolor import DDColor, ColorizationPipeline


@dataclass
class DDColorPILPipeline:
    model_name: Literal[
        "ddcolor_modelscope",
        "ddcolor_paper",
        "ddcolor_artistic",
        "ddcolor_paper_tiny",
    ] = "ddcolor_modelscope"
    input_size: int = 512
    device: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)

        repo_id = self.model_name
        if "/" not in repo_id:
            repo_id = f"piddnad/{repo_id}"

        self.model = DDColorHF.from_pretrained(repo_id)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.colorizer = ColorizationPipeline(
            self.model,
            input_size=self.input_size,
            device=torch.device(self.device),
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Input:  PIL.Image.Image
        Output: PIL.Image.Image (RGB, colorized)
        """
        # Normalize PIL input to RGB
        img_rgb = img.convert("RGB")

        # PIL RGB -> numpy RGB -> OpenCV BGR
        rgb = np.array(img_rgb, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # DDColor inference
        out_bgr = self.colorizer.process(bgr)

        # OpenCV BGR -> PIL RGB
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb, mode="RGB")

if __name__ == "__main__":
    pipe = DDColorPILPipeline(
        model_name="ddcolor_modelscope",
    )

    img = Image.open("images/IMG_0956_sklearn_grayscale.JPEG")
    colorized = pipe(img)
    print("Finished colorizing the image!")
    print(colorized)
    save_path = "IMG_0956_ddcolor.JPEG"
    print(f"Saving to {save_path}")
    colorized.save(save_path)
    # colorized.show()
