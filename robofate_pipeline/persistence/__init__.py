"""Original teacher persistence and camera budget, with explicit external roots."""
from types import SimpleNamespace

from . import camera, files, native_files
from .camera_budget import CameraBudget
from .journal import Journal
from .plain import plain


def api():
    return SimpleNamespace(camera=camera, peer=native_files, host=files, plain=plain)
