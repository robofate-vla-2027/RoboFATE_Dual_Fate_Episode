"""Original production teacher with explicitly bound, isolated runtime resources."""

from .runtime import install
from .specification import model, specification

__all__ = ["install", "model", "specification"]
