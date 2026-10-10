"""Private linear solves for differential kinematics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._constraints import acceleration_rhs, jacobian, _finite_driver_sample
from ._drivers import normalize_drivers
from .driver import KinematicDriver
from ._scaling import NumericalScaling
from .diagnostics import _rank_analysis
from .errors import KinematicSolveError, SolveFailureContext
from .joints import PrismaticJoint, RevoluteJoint
from .model import Link, Mechanism

_Joint = RevoluteJoint | PrismaticJoint
_LINEAR_RESIDUAL_TOL = 1e-9


def solve_velocity(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver | Sequence[KinematicDriver],
    q: np.ndarray,
    driver_value: float | Sequence[float] | np.ndarray,
    driver_velocity: float | Sequence[float] | np.ndarray,
    scaling: NumericalScaling,
    *,
    sample_index: int | None = None,
) -> np.ndarray:
    """Solve one generalized velocity state from differentiated constraints."""
    velocity = _finite_driver_sample(
        driver_velocity, len(normalize_drivers(driver)), name="driver_velocity"
    )
    matrix = jacobian(mechanism, links, joints, driver, q, driver_value)
    rhs = np.zeros(matrix.shape[0], dtype=float)
    rhs[-len(velocity):] = velocity
    return _solve_linear_state(
        scaling.scale_jacobian(matrix),
        scaling.scale_rhs(rhs),
        scaling=scaling,
        link_count=len(links),
        joint_row_count=2 * len(joints),
        stage="velocity",
        driver_value=driver_value,
        sample_index=sample_index,
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
    sample_index: int | None = None,
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
        joint_row_count=2 * len(joints),
        stage="driver tangent",
        driver_value=driver_value,
        sample_index=sample_index,
    )


def solve_acceleration(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver | Sequence[KinematicDriver],
    q: np.ndarray,
    q_dot: np.ndarray,
    driver_value: float | Sequence[float] | np.ndarray,
    driver_acceleration: float | Sequence[float] | np.ndarray,
    scaling: NumericalScaling,
    *,
    sample_index: int | None = None,
) -> np.ndarray:
    """Solve one generalized acceleration state from second-order constraints."""
    prescribed = _finite_driver_sample(
        driver_acceleration, len(normalize_drivers(driver)), name="driver_acceleration"
    )
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
        joint_row_count=2 * len(joints),
        stage="acceleration",
        driver_value=driver_value,
        sample_index=sample_index,
    )


def _solve_linear_state(
    matrix_hat: np.ndarray,
    rhs_hat: np.ndarray,
    *,
    scaling: NumericalScaling,
    link_count: int,
    joint_row_count: int,
    stage: str,
    driver_value: float | Sequence[float] | np.ndarray,
    sample_index: int | None,
) -> np.ndarray:
    """Solve only when geometric constraints and prescribed inputs are regular.

    A numerically singular Jacobian cannot provide unique differential states;
    a nonconverged position iterate is handled separately by the corrector.
    """
    condition, minimum, rank, joint_rank, issue = _rank_analysis(
        matrix_hat, joint_row_count=joint_row_count
    )

    def context(kind: str, *, residual: float | None = None) -> SolveFailureContext:
        return SolveFailureContext(
            stage=stage,
            sample_index=sample_index,
            driver_positions=_driver_positions_tuple(driver_value),
            failure_kind=kind,
            rank_issue=issue,
            joint_rank=joint_rank,
            condition_number=condition,
            min_singular_value=minimum,
            rank=rank,
            residual_norm=residual,
        )

    if issue != "regular":
        description = (
            "joint constraint Jacobian loses row rank"
            if issue == "joint_rank_loss"
            else "prescribed drivers do not determine all local motions"
        )
        raise KinematicSolveError(
            _failure_message(
                stage, driver_value, sample_index=sample_index,
                residual_norm=float("nan"), reason=description,
            ),
            context=context(issue),
        )

    try:
        candidate_hat = np.asarray(np.linalg.solve(matrix_hat, rhs_hat), dtype=float)
    except np.linalg.LinAlgError as error:
        raise KinematicSolveError(
            _failure_message(
                stage, driver_value, sample_index=sample_index,
                residual_norm=float("nan"), reason=f"linear solve failed: {error}",
            ),
            context=context("linear_failure"),
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
            stage, driver_value, sample_index=sample_index,
            residual_norm=residual_norm,
            reason="invalid or inaccurate linear solution",
        ),
        context=context(
            "linear_failure",
            residual=residual_norm if np.isfinite(residual_norm) else None,
        ),
    )


def _failure_message(
    stage: str,
    driver_value: float | Sequence[float] | np.ndarray,
    *,
    sample_index: int | None,
    residual_norm: float,
    reason: str,
) -> str:
    formatted_value = _format_driver_values(driver_value)
    location = (
        f"sample index {sample_index} (value={formatted_value})"
        if sample_index is not None
        else f"driver value {formatted_value}"
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



def _format_driver_values(value: float | Sequence[float] | np.ndarray) -> str:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return f"{float(array):.12g}"
    return "[" + ", ".join(f"{float(item):.12g}" for item in array) + "]"


def _driver_positions_tuple(
    value: float | Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    values = np.atleast_1d(np.asarray(value, dtype=float))
    return tuple(float(item) for item in values)
