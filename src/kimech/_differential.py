"""Private linear solves for differential kinematics."""

from __future__ import annotations

import numpy as np

from ._constraints import acceleration_rhs, jacobian
from ._scaling import NumericalScaling
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
    scaling: NumericalScaling,
    *,
    input_index: int | None = None,
) -> np.ndarray:
    """Solve one generalized velocity state from the differentiated constraints."""
    velocity = _finite_scalar(input_velocity, name="input_velocity")
    matrix = jacobian(mechanism, links, joints, input_joint, q, input_value)
    rhs = np.zeros(matrix.shape[0], dtype=float)
    rhs[-1] = velocity
    return _solve_linear_state(
        scaling.scale_jacobian(matrix),
        scaling.scale_rhs(rhs),
        scaling=scaling,
        link_count=len(links),
        stage="velocity",
        input_value=input_value,
        input_index=input_index,
    )


def solve_input_tangent(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    q: np.ndarray,
    input_value: float,
    scaling: NumericalScaling,
    *,
    input_index: int | None = None,
) -> np.ndarray:
    """Solve the configuration tangent dq/du for continuation.

    The differentiated constraint system is J(q) dq/du = e_driver because
    the prescribed driver equation is c(q) - u = 0. The returned state
    derivative is with respect to the input coordinate, not physical time.
    """
    matrix = jacobian(mechanism, links, joints, input_joint, q, input_value)
    rhs = np.zeros(matrix.shape[0], dtype=float)
    rhs[-1] = 1.0
    return _solve_linear_state(
        scaling.scale_jacobian(matrix),
        scaling.scale_rhs(rhs),
        scaling=scaling,
        link_count=len(links),
        stage="input tangent",
        input_value=input_value,
        input_index=input_index,
    )

def solve_acceleration(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    q: np.ndarray,
    q_dot: np.ndarray,
    input_value: float,
    input_acceleration: float,
    scaling: NumericalScaling,
    *,
    input_index: int | None = None,
) -> np.ndarray:
    """Solve one generalized acceleration state from second-order constraints."""
    prescribed = _finite_scalar(input_acceleration, name="input_acceleration")
    matrix = jacobian(mechanism, links, joints, input_joint, q, input_value)
    rhs = acceleration_rhs(
        mechanism,
        links,
        joints,
        input_joint,
        q,
        q_dot,
        input_value,
        prescribed,
    )
    return _solve_linear_state(
        scaling.scale_jacobian(matrix),
        scaling.scale_rhs(rhs),
        scaling=scaling,
        link_count=len(links),
        stage="acceleration",
        input_value=input_value,
        input_index=input_index,
    )


def _solve_linear_state(
    matrix_hat: np.ndarray,
    rhs_hat: np.ndarray,
    *,
    scaling: NumericalScaling,
    link_count: int,
    stage: str,
    input_value: float,
    input_index: int | None,
) -> np.ndarray:
    try:
        candidate_hat = np.asarray(np.linalg.solve(matrix_hat, rhs_hat), dtype=float)
    except np.linalg.LinAlgError as error:
        raise KinematicSolveError(
            _failure_message(
                stage,
                input_value,
                input_index=input_index,
                residual_norm=float("nan"),
                reason=f"linear solve failed: {error}",
            )
        ) from error

    expected_shape = (3 * link_count,)
    candidate_valid = (
        candidate_hat.shape == expected_shape and np.all(np.isfinite(candidate_hat))
    )
    residual_norm = float("nan")
    if candidate_valid:
        linear_residual_hat = matrix_hat @ candidate_hat - rhs_hat
        residual_norm = float(np.linalg.norm(linear_residual_hat, ord=np.inf))

    if (
        candidate_valid
        and np.isfinite(residual_norm)
        and residual_norm <= _LINEAR_RESIDUAL_TOL
    ):
        return scaling.unscale_state(candidate_hat).copy()

    raise KinematicSolveError(
        _failure_message(
            stage,
            input_value,
            input_index=input_index,
            residual_norm=residual_norm,
            reason="invalid or inaccurate linear solution",
        )
    )


def _failure_message(
    stage: str,
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
        f"failed to solve {stage} at {location}: residual_inf={norm_text}; "
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
