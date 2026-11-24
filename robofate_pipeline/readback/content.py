"""Deployed constant-RGB diagnostics; freshness is not inferred from contrast."""
import hashlib
from ..evidence import need, file_ref
from pathlib import Path

def policy_reference():
    return file_ref(Path(__file__).resolve())

def validate_rgb(image, stats):
    import numpy as np
    need(isinstance(image, np.ndarray) and image.dtype == np.uint8
         and image.shape == (480, 640, 3), "Malformed or wrong-size actual RGB")
    need(stats["shape"] == list(image.shape)
         and stats["sha256"] == hashlib.sha256(image.tobytes()).hexdigest(),
         "Actual RGB and saved statistics differ")
    constant = bool(image.std() == 0)
    if "constant_pixel_values" in stats:
        need(type(stats["constant_pixel_values"]) is bool
             and stats["constant_pixel_values"] == constant,
             "Saved constant-pixel diagnostic differs from actual RGB")
    return {"constant_pixel_values": constant, "std": float(image.std())}
