"""Structured numerical diagnostics for kinematic solutions."""

from __future__ import annotations

import numpy as np


class SolveDiagnostics:
    """Numerical diagnostics recorded for an ordered kinematic solution.

    Jacobian metrics refer to Kimech's dimensionless scaled constraint
    Jacobian, so they are intended to be comparable across consistent choices
    of linear units.
    """

    __slots__ = (
        "_condition_numbers",
        "_min_singular_values",
        "_ranks",
        "_subdivision_counts",
    )

    def __init__(
        self,
        condition_numbers,
        min_singular_values,
        ranks,
        *,
        subdivision_counts=None,
    ) -> None:
        condition = _condition_array(condition_numbers)
        minimum = _finite_float_array(
            min_singular_values,
            name="min_singular_values",
        )
        rank = _rank_array(ranks)
        subdivisions = (
            None
            if subdivision_counts is None
            else _nonnegative_int_array(
                subdivision_counts,
                name="subdivision_counts",
            )
        )

        if minimum.shape != condition.shape or rank.shape != condition.shape:
            raise ValueError("all diagnostic histories must have the same shape")
        if subdivisions is not None and subdivisions.shape != condition.shape:
            raise ValueError("all diagnostic histories must have the same shape")

        self._condition_numbers = condition
        self._min_singular_values = minimum
        self._ranks = rank
        self._subdivision_counts = subdivisions

    @property
    def condition_numbers(self) -> np.ndarray:
        """Return scaled-Jacobian condition numbers."""
        return self._condition_numbers.copy()

    @property
    def min_singular_values(self) -> np.ndarray:
        """Return the smallest singular value of the scaled Jacobian."""
        return self._min_singular_values.copy()

    @property
    def ranks(self) -> np.ndarray:
        """Return numerical ranks of the scaled Jacobian."""
        return self._ranks.copy()

    @property
    def subdivision_counts(self) -> np.ndarray | None:
        """Return accepted internal subdivision counts for each requested sample."""
        if self._subdivision_counts is None:
            return None
        return self._subdivision_counts.copy()

    def __len__(self) -> int:
        return len(self._condition_numbers)

    def _slice(self, index: slice) -> SolveDiagnostics:
        """Return a sliced diagnostics history for result-container internals."""
        if not isinstance(index, slice):
            raise TypeError("index must be a slice")
        return SolveDiagnostics(
            self._condition_numbers[index],
            self._min_singular_values[index],
            self._ranks[index],
            subdivision_counts=(
                None
                if self._subdivision_counts is None
                else self._subdivision_counts[index]
            ),
        )


def _condition_array(value: object) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("condition_numbers must be numeric") from error
    if array.ndim != 1:
        raise ValueError("condition_numbers must be 1-dimensional")
    if np.any(np.isnan(array)) or np.any(array < 1.0):
        raise ValueError(
            "condition_numbers must contain values greater than or equal to 1 or inf"
        )
    return array.copy()


def _finite_float_array(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array.copy()


def _rank_array(value: object) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("ranks must be integer-valued") from error
    if array.ndim != 1:
        raise ValueError("ranks must be 1-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("ranks must be integer-valued")
    result = array.astype(int, copy=True)
    if np.any(result < 0):
        raise ValueError("ranks must contain only non-negative values")
    return result



def _nonnegative_int_array(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be integer-valued") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must be integer-valued")
    result = array.astype(int, copy=True)
    if np.any(result < 0):
        raise ValueError(f"{name} must contain only non-negative values")
    return result

def _jacobian_metrics(matrix: object) -> tuple[float, float, int]:
    """Return condition number, smallest singular value, and numerical rank."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")

    singular_values = np.linalg.svd(values, compute_uv=False)
    if singular_values.size == 0:
        return float("inf"), 0.0, 0

    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    tolerance = max(values.shape) * np.finfo(float).eps * sigma_max
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = float("inf") if sigma_min == 0.0 else sigma_max / sigma_min
    return condition, sigma_min, rank
