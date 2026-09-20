"""Structured numerical diagnostics for kinematic solutions."""

from __future__ import annotations

import operator

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
    )

    def __init__(
        self,
        condition_numbers,
        min_singular_values,
        ranks,
    ) -> None:
        condition = _condition_array(condition_numbers)
        minimum = _finite_float_array(
            min_singular_values,
            name="min_singular_values",
        )
        rank = _rank_array(ranks)

        if minimum.shape != condition.shape or rank.shape != condition.shape:
            raise ValueError("all diagnostic histories must have the same shape")

        self._condition_numbers = condition
        self._min_singular_values = minimum
        self._ranks = rank

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

    def __len__(self) -> int:
        return len(self._condition_numbers)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return SolveDiagnostics(
                self._condition_numbers[index],
                self._min_singular_values[index],
                self._ranks[index],
            )

        item = operator.index(index)
        return {
            "condition_number": float(self._condition_numbers[item]),
            "min_singular_value": float(self._min_singular_values[item]),
            "rank": int(self._ranks[item]),
        }


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
