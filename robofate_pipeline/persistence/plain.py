"""Original JSON projection; never serializes arbitrary objects."""
import numpy as np
from ..recording.takeover import require

def plain(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, 'Object action/observation')
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value
