# GTA1/src/qwen_vl_utils.py
# Minimal utils compatible with Qwen2.5-VL examples.

import math
from typing import Tuple, List, Dict, Any

def _round_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int(max(m, int(round(x / m)) * m))

def smart_resize(
    h: int,
    w: int,
    factor: int = 32,
    min_pixels: int = 3136,         # 56*56 default in Qwen examples
    max_pixels: int = 4096 * 2160,  # cap to something reasonable
) -> Tuple[int, int]:
    """
    Compute a (rh, rw) that:
      - preserves aspect ratio,
      - keeps rh*rw within [min_pixels, max_pixels],
      - and makes both rh and rw multiples of `factor`.
    """
    assert h > 0 and w > 0, "Input image size must be positive"
    area = h * w

    # Desired scale so that area*s^2 is within [min_pixels, max_pixels]
    s_min = math.sqrt(min_pixels / area) if area < min_pixels else 1.0
    s_max = math.sqrt(max_pixels / area) if area > max_pixels else 1.0

    # Choose s that brings area closest to within bounds, preferring no scale if already valid
    if area < min_pixels:
        s = s_min
    elif area > max_pixels:
        s = s_max
    else:
        s = 1.0

    rh = max(1, int(round(h * s)))
    rw = max(1, int(round(w * s)))

    # Round to multiples of factor, but keep aspect ratio roughly
    rh = _round_to_multiple(rh, factor)
    rw = _round_to_multiple(rw, factor)

    # Final clamp if rounding pushed us out of bounds
    # Try to shrink if we exceeded max_pixels
    if rh * rw > max_pixels:
        shrink = math.sqrt(max_pixels / (rh * rw))
        rh = _round_to_multiple(max(1, int(rh * shrink)), factor)
        rw = _round_to_multiple(max(1, int(rw * shrink)), factor)

    # Try to grow if we are far below min_pixels
    if rh * rw < min_pixels:
        grow = math.sqrt(min_pixels / (rh * rw))
        rh = _round_to_multiple(max(1, int(rh * grow)), factor)
        rw = _round_to_multiple(max(1, int(rw * grow)), factor)

    return int(rh), int(rw)

# Some GTA/Qwen example code imports this symbol during inference.
# Keep a no-op shim so imports don't fail in training runs.
def process_vision_info(messages: List[Dict[str, Any]]):
    """
    Stub for compatibility with Qwen example code.
    Returns (images, videos) lists expected by AutoProcessor if you ever use it.
    Training path in GTA1 typically doesn't need this.
    """
    image_inputs, video_inputs = [], []
    for m in messages:
        if not isinstance(m.get("content"), list):
            continue
        for chunk in m["content"]:
            if chunk.get("type") == "image":
                image_inputs.append(chunk.get("image"))
            if chunk.get("type") == "video":
                video_inputs.append(chunk.get("video"))
    return image_inputs, video_inputs
