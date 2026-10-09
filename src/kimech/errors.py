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
    driver_index: int | None = None
    driver_position: float | None = None
    sample_index: int | None = None
    driver_positions: tuple[float, ...] | None = None
    failure_kind: str | None = None
    joint_rank: int | None = None
    rank_issue: str | None = None
    residual_norm: float | None = None
    condition_number: float | None = None
    min_singular_value: float | None = None
    rank: int | None = None
    attempted_strategies: tuple[str, ...] = ()
    corrector_attempts: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")

        if self.driver_index is not None:
            if not isinstance(self.driver_index, int) or self.driver_index < 0:
                raise ValueError("driver_index must be a non-negative integer or None")

        if self.sample_index is not None:
            if not isinstance(self.sample_index, int) or self.sample_index < 0:
                raise ValueError("sample_index must be a non-negative integer or None")
        if (self.sample_index is not None and self.driver_index is not None
                and self.sample_index != self.driver_index):
            raise ValueError("sample_index and driver_index must refer to the same sample")

        if self.driver_positions is not None:
            if (not isinstance(self.driver_positions, tuple)
                    or not self.driver_positions
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in self.driver_positions)):
                raise ValueError("driver_positions must be a nonempty tuple of finite numbers")

        if self.failure_kind is not None and self.failure_kind not in (
            "nonconvergence", "joint_rank_loss", "dependent_drivers", "linear_failure"
        ):
            raise ValueError("failure_kind is not a recognized failure classification")
        if self.rank_issue is not None and self.rank_issue not in (
            "regular", "joint_rank_loss", "dependent_drivers"
        ):
            raise ValueError("rank_issue is not a recognized rank classification")
        if self.joint_rank is not None:
            if not isinstance(self.joint_rank, int) or self.joint_rank < 0:
                raise ValueError("joint_rank must be a non-negative integer or None")

        if self.driver_position is not None and not math.isfinite(self.driver_position):
            raise ValueError("driver_position must be finite or None")

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
