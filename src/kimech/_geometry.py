"""Private planar-geometry helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def as_vector2(value: Sequence[float], *, name: str = "vector") -> tuple[float, float]:
    """Validate and normalize an input as a finite 2D vector tuple."""
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly two coordinates")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return float(array[0]), float(array[1])


def normalize_axis(value: Sequence[float], *, name: str = "axis") -> tuple[float, float]:
    """Return a normalized finite 2D axis."""
    x, y = as_vector2(value, name=name)
    norm = float(np.hypot(x, y))
    if np.isclose(norm, 0.0):
        raise ValueError(f"{name} must have non-zero length")
    return x / norm, y / norm


def rotation_matrix(theta: float) -> np.ndarray:
    """Return the 2×2 planar rotation matrix for ``theta`` radians."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def perpendicular(vector: Sequence[float]) -> np.ndarray:
    """Return the +90° perpendicular of a 2D vector."""
    x, y = as_vector2(vector)
    return np.array([-y, x], dtype=float)
