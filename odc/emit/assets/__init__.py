"""
Binary assets management
"""

from pathlib import Path

import numpy as np

__all__ = (
    "EMIT_WAVELENGTH_VALUES",
    "EMIT_WAVELENGTH_BYTES",
)


def _load_np(name: str) -> np.ndarray:
    return np.load(Path(__file__).parent / "coords.npz")[name]


def _load_list(name: str) -> list[float]:
    return _load_np(name).tolist()


EMIT_WAVELENGTH_VALUES = _load_list("wavelengths")
EMIT_WAVELENGTH_BYTES = _load_np("wavelengths").tobytes()
