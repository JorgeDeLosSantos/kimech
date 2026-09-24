"""Structured numerical diagnostics for kinematic solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SolveDiagnosticSummary:
    """Compact descriptive summary of a solve-diagnostics history."""

    sample_count: int
    worst_condition_index: int | None
    worst_condition_number: float | None
    minimum_singular_value_index: int | None
    minimum_singular_value: float | None
    minimum_rank: int | None
    max_subdivision_index: int | None
    max_subdivision_count: int | None
    max_corrector_attempt_index: int | None
    max_corrector_attempts: int | None
    strategy_counts: tuple[tuple[str, int], ...]


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
        "_strategies",
        "_corrector_attempts",
        "_residual_norms",
    )

    def __init__(
        self,
        condition_numbers,
        min_singular_values,
        ranks,
        *,
        subdivision_counts=None,
        strategies=None,
        corrector_attempts=None,
        residual_norms=None,
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
        strategy = (
            None
            if strategies is None
            else _strategy_array(strategies)
        )
        attempts = (
            None
            if corrector_attempts is None
            else _nonnegative_int_array(
                corrector_attempts,
                name="corrector_attempts",
            )
        )
        residual_history = (
            None
            if residual_norms is None
            else _finite_float_array(
                residual_norms,
                name="residual_norms",
            )
        )

        if minimum.shape != condition.shape or rank.shape != condition.shape:
            raise ValueError("all diagnostic histories must have the same shape")
        for optional in (subdivisions, strategy, attempts, residual_history):
            if optional is not None and optional.shape != condition.shape:
                raise ValueError("all diagnostic histories must have the same shape")

        self._condition_numbers = condition
        self._min_singular_values = minimum
        self._ranks = rank
        self._subdivision_counts = subdivisions
        self._strategies = strategy
        self._corrector_attempts = attempts
        self._residual_norms = residual_history

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

    @property
    def strategies(self) -> np.ndarray | None:
        """Return the accepted position-solve strategy for each requested sample."""
        if self._strategies is None:
            return None
        return self._strategies.copy()

    @property
    def corrector_attempts(self) -> np.ndarray | None:
        """Return nonlinear position-corrector attempts per requested sample."""
        if self._corrector_attempts is None:
            return None
        return self._corrector_attempts.copy()

    @property
    def residual_norms(self) -> np.ndarray | None:
        """Return final scaled position residual infinity norms."""
        if self._residual_norms is None:
            return None
        return self._residual_norms.copy()

    def __len__(self) -> int:
        return len(self._condition_numbers)

    def summary(self) -> SolveDiagnosticSummary:
        """Return descriptive extrema and solve-effort counts for this history."""
        count = len(self)
        if count == 0:
            return SolveDiagnosticSummary(
                sample_count=0,
                worst_condition_index=None,
                worst_condition_number=None,
                minimum_singular_value_index=None,
                minimum_singular_value=None,
                minimum_rank=None,
                max_subdivision_index=None,
                max_subdivision_count=None,
                max_corrector_attempt_index=None,
                max_corrector_attempts=None,
                strategy_counts=(),
            )

        worst_condition_index = int(np.argmax(self._condition_numbers))
        minimum_singular_value_index = int(np.argmin(self._min_singular_values))

        if self._subdivision_counts is None:
            max_subdivision_index = None
            max_subdivision_count = None
        else:
            max_subdivision_index = int(np.argmax(self._subdivision_counts))
            max_subdivision_count = int(
                self._subdivision_counts[max_subdivision_index]
            )

        if self._corrector_attempts is None:
            max_corrector_attempt_index = None
            max_corrector_attempts = None
        else:
            max_corrector_attempt_index = int(np.argmax(self._corrector_attempts))
            max_corrector_attempts = int(
                self._corrector_attempts[max_corrector_attempt_index]
            )

        if self._strategies is None:
            strategy_counts: tuple[tuple[str, int], ...] = ()
        else:
            strategy_counts = tuple(
                (strategy, int(np.count_nonzero(self._strategies == strategy)))
                for strategy in _STRATEGY_ORDER
                if np.any(self._strategies == strategy)
            )

        return SolveDiagnosticSummary(
            sample_count=count,
            worst_condition_index=worst_condition_index,
            worst_condition_number=float(
                self._condition_numbers[worst_condition_index]
            ),
            minimum_singular_value_index=minimum_singular_value_index,
            minimum_singular_value=float(
                self._min_singular_values[minimum_singular_value_index]
            ),
            minimum_rank=int(np.min(self._ranks)),
            max_subdivision_index=max_subdivision_index,
            max_subdivision_count=max_subdivision_count,
            max_corrector_attempt_index=max_corrector_attempt_index,
            max_corrector_attempts=max_corrector_attempts,
            strategy_counts=strategy_counts,
        )

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
            strategies=(
                None
                if self._strategies is None
                else self._strategies[index]
            ),
            corrector_attempts=(
                None
                if self._corrector_attempts is None
                else self._corrector_attempts[index]
            ),
            residual_norms=(
                None
                if self._residual_norms is None
                else self._residual_norms[index]
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


_STRATEGY_ORDER = (
    "initial_guess",
    "predictor",
    "warm_start",
    "subdivision",
)
_ALLOWED_STRATEGIES = frozenset(_STRATEGY_ORDER)


def _strategy_array(value: object) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=str)
    except (TypeError, ValueError) as error:
        raise TypeError("strategies must be string-valued") from error
    if array.ndim != 1:
        raise ValueError("strategies must be 1-dimensional")
    if not all(item in _ALLOWED_STRATEGIES for item in array):
        allowed = ", ".join(sorted(_ALLOWED_STRATEGIES))
        raise ValueError(f"strategies must contain only: {allowed}")
    return array.copy()

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
