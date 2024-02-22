"""
Binary assets management
"""

from pathlib import Path

import numpy as np

__all__ = ("EMIT_WAVELENGTH_VALUES",)


def _load(name: str) -> list[float]:
    return np.load(Path(__file__).parent / "coords.npz")[name].tolist()


EMIT_WAVELENGTH_VALUES = _load("wavelengths")
