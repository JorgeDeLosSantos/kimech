"""Public exceptions raised by Kimech."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SolveFailureContext:
    """Structured context attached to a kinematic solve failure.

    Fields are descriptive and may be unavailable when a failure occurs before
    a reliable candidate state or Jacobian can be evaluated.
    """

    stage: str
    input_index: int | None = None
    input_position: float | None = None
    residual_norm: float | None = None
    condition_number: float | None = None
    min_singular_value: float | None = None
    rank: int | None = None
    attempted_strategies: tuple[str, ...] = ()
    corrector_attempts: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")

        if self.input_index is not None:
            if not isinstance(self.input_index, int) or self.input_index < 0:
                raise ValueError("input_index must be a non-negative integer or None")

        if self.input_position is not None and not math.isfinite(self.input_position):
            raise ValueError("input_position must be finite or None")

        if self.residual_norm is not None:
            if not math.isfinite(self.residual_norm) or self.residual_norm < 0.0:
                raise ValueError("residual_norm must be finite and non-negative or None")

        if self.condition_number is not None:
            if math.isnan(self.condition_number) or self.condition_number < 1.0:
                raise ValueError(
                    "condition_number must be greater than or equal to 1, inf, or None"
                )

        if self.min_singular_value is not None:
            if (
                not math.isfinite(self.min_singular_value)
                or self.min_singular_value < 0.0
            ):
                raise ValueError(
                    "min_singular_value must be finite and non-negative or None"
                )

        if self.rank is not None:
            if not isinstance(self.rank, int) or self.rank < 0:
                raise ValueError("rank must be a non-negative integer or None")

        if not isinstance(self.attempted_strategies, tuple) or not all(
            isinstance(item, str) and item
            for item in self.attempted_strategies
        ):
            raise TypeError("attempted_strategies must be a tuple of non-empty strings")

        if self.corrector_attempts is not None:
            if (
                not isinstance(self.corrector_attempts, int)
                or self.corrector_attempts < 0
            ):
                raise ValueError(
                    "corrector_attempts must be a non-negative integer or None"
                )


class KimechError(Exception):
    """Base class for Kimech-specific errors."""


class InvalidModelError(KimechError):
    """Raised when a mechanism or solve problem is structurally invalid."""


class KinematicSolveError(KimechError):
    """Raised when a kinematic configuration cannot be solved reliably."""

    def __init__(
        self,
        message: str,
        *,
        context: SolveFailureContext | None = None,
    ) -> None:
        super().__init__(message)
        if context is not None and not isinstance(context, SolveFailureContext):
            raise TypeError("context must be a SolveFailureContext or None")
        self.context = context
