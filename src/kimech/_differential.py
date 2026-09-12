"""Private linear solves for differential kinematics."""

from __future__ import annotations

import numpy as np

from ._constraints import jacobian
from .errors import KinematicSolveError
from .joints import PrismaticJoint, RevoluteJoint
from .model import Link, Mechanism

_Joint = RevoluteJoint | PrismaticJoint
_LINEAR_RESIDUAL_TOL = 1e-9


def solve_velocity(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    q: np.ndarray,
    input_value: float,
    input_velocity: float,
    *,
    input_index: int | None = None,
) -> np.ndarray:
    """Solve one generalized velocity state from the differentiated constraints."""
    velocity = _finite_scalar(input_velocity, name="input_velocity")
    matrix = jacobian(mechanism, links, joints, input_joint, q, input_value)
    rhs = np.zeros(matrix.shape[0], dtype=float)
    rhs[-1] = velocity

    try:
        candidate = np.asarray(np.linalg.solve(matrix, rhs), dtype=float)
    except np.linalg.LinAlgError as error:
        raise KinematicSolveError(
            _failure_message(
                input_value,
                input_index=input_index,
                residual_norm=float("nan"),
                reason=f"linear solve failed: {error}",
            )
        ) from error

    expected_shape = (3 * len(links),)
    candidate_valid = candidate.shape == expected_shape and np.all(np.isfinite(candidate))
    residual_norm = float("nan")
    if candidate_valid:
        linear_residual = matrix @ candidate - rhs
        residual_norm = float(np.linalg.norm(linear_residual, ord=np.inf))

    if (
        candidate_valid
        and np.isfinite(residual_norm)
        and residual_norm <= _LINEAR_RESIDUAL_TOL
    ):
        return candidate.copy()

    raise KinematicSolveError(
        _failure_message(
            input_value,
            input_index=input_index,
            residual_norm=residual_norm,
            reason="invalid or inaccurate linear solution",
        )
    )


def _failure_message(
    input_value: float,
    *,
    input_index: int | None,
    residual_norm: float,
    reason: str,
) -> str:
    location = (
        f"input index {input_index} (value={input_value:.12g})"
        if input_index is not None
        else f"input value {input_value:.12g}"
    )
    norm_text = f"{residual_norm:.12g}" if np.isfinite(residual_norm) else "unavailable"
    return (
        f"failed to solve velocity at {location}: residual_inf={norm_text}; "
        f"reason={reason}"
    )


def _finite_scalar(value: object, *, name: str) -> float:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric scalar") from error
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
