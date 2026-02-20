import numpy as np
from PIL import Image

def image_to_heightmap(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img)

    height_map = (
        0.7 * rgb[:, :, 0] +
        0.2 * rgb[:, :, 1] +
        0.0 * rgb[:, :, 2]
    ).astype(np.uint8)

    denom = (height_map.max() - height_map.min())
    if denom == 0:
        return np.zeros_like(height_map, dtype=np.uint8)

    hm = ((height_map - height_map.min()) / denom * 255).astype(np.uint8)
    return hm
