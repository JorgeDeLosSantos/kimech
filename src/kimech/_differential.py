"""Private linear solves for differential kinematics."""

from __future__ import annotations

import numpy as np

from ._constraints import acceleration_rhs, jacobian
from .driver import KinematicDriver
from ._scaling import NumericalScaling
from .diagnostics import _jacobian_metrics
from .errors import KinematicSolveError, SolveFailureContext
from .joints import PrismaticJoint, RevoluteJoint
from .model import Link, Mechanism

_Joint = RevoluteJoint | PrismaticJoint
_LINEAR_RESIDUAL_TOL = 1e-9


def solve_velocity(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    q: np.ndarray,
    driver_value: float,
    driver_velocity: float,
    scaling: NumericalScaling,
    *,
    driver_index: int | None = None,
) -> np.ndarray:
    """Solve one generalized velocity state from differentiated constraints."""
    velocity = _finite_scalar(driver_velocity, name="driver_velocity")
    matrix = jacobian(mechanism, links, joints, driver, q, driver_value)
    rhs = np.zeros(matrix.shape[0], dtype=float)
    rhs[-1] = velocity
    return _solve_linear_state(
        scaling.scale_jacobian(matrix),
        scaling.scale_rhs(rhs),
        scaling=scaling,
        link_count=len(links),
        stage="velocity",
        driver_value=driver_value,
        driver_index=driver_index,
    )


def solve_driver_tangent(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    q: np.ndarray,
    driver_value: float,
    scaling: NumericalScaling,
    *,
    driver_index: int | None = None,
) -> np.ndarray:
    """Solve the configuration tangent dq/du for continuation."""
    matrix = jacobian(mechanism, links, joints, driver, q, driver_value)
    rhs = np.zeros(matrix.shape[0], dtype=float)
    rhs[-1] = 1.0
    return _solve_linear_state(
        scaling.scale_jacobian(matrix),
        scaling.scale_rhs(rhs),
        scaling=scaling,
        link_count=len(links),
        stage="driver tangent",
        driver_value=driver_value,
        driver_index=driver_index,
    )


def solve_acceleration(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    q: np.ndarray,
    q_dot: np.ndarray,
    driver_value: float,
    driver_acceleration: float,
    scaling: NumericalScaling,
    *,
    driver_index: int | None = None,
) -> np.ndarray:
    """Solve one generalized acceleration state from second-order constraints."""
    prescribed = _finite_scalar(driver_acceleration, name="driver_acceleration")
    matrix = jacobian(mechanism, links, joints, driver, q, driver_value)
    rhs = acceleration_rhs(
        mechanism,
        links,
        joints,
        driver,
        q,
        q_dot,
        driver_value,
        prescribed,
    )
    return _solve_linear_state(
        scaling.scale_jacobian(matrix),
        scaling.scale_rhs(rhs),
        scaling=scaling,
        link_count=len(links),
        stage="acceleration",
        driver_value=driver_value,
        driver_index=driver_index,
    )


def _solve_linear_state(
    matrix_hat: np.ndarray,
    rhs_hat: np.ndarray,
    *,
    scaling: NumericalScaling,
    link_count: int,
    stage: str,
    driver_value: float,
    driver_index: int | None,
) -> np.ndarray:
    try:
        candidate_hat = np.asarray(np.linalg.solve(matrix_hat, rhs_hat), dtype=float)
    except np.linalg.LinAlgError as error:
        condition, minimum, rank = _jacobian_metrics(matrix_hat)
        raise KinematicSolveError(
            _failure_message(
                stage,
                driver_value,
                driver_index=driver_index,
                residual_norm=float("nan"),
                reason=f"linear solve failed: {error}",
            ),
            context=SolveFailureContext(
                stage=stage,
                driver_index=driver_index,
                driver_position=driver_value,
                condition_number=condition,
                min_singular_value=minimum,
                rank=rank,
            ),
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

    condition, minimum, rank = _jacobian_metrics(matrix_hat)
    raise KinematicSolveError(
        _failure_message(
            stage,
            driver_value,
            driver_index=driver_index,
            residual_norm=residual_norm,
            reason="invalid or inaccurate linear solution",
        ),
        context=SolveFailureContext(
            stage=stage,
            driver_index=driver_index,
            driver_position=driver_value,
            residual_norm=(
                residual_norm if np.isfinite(residual_norm) else None
            ),
            condition_number=condition,
            min_singular_value=minimum,
            rank=rank,
        ),
    )


def _failure_message(
    stage: str,
    driver_value: float,
    *,
    driver_index: int | None,
    residual_norm: float,
    reason: str,
) -> str:
    location = (
        f"driver index {driver_index} (value={driver_value:.12g})"
        if driver_index is not None
        else f"driver value {driver_value:.12g}"
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
