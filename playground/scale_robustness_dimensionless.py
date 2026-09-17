"""Compare Kimech's current position solve with a dimensionless prototype.

This is the second 0.3.0 numerical-robustness study.  It intentionally leaves
``kimech.solve`` unchanged and wraps the existing residual/Jacobian in diagonal
coordinate and equation scalings:

    q = D_q q_hat
    Phi_hat = D_phi^{-1} Phi
    J_hat = D_phi^{-1} J D_q

For each mechanism family and geometric scale used by ``scale_robustness.py``,
the script reports side-by-side diagnostics for the current/raw formulation and
the dimensionless formulation.

Run from the repository root::

    python playground/scale_robustness_dimensionless.py

The characteristic length is still supplied by each study case.  Inferring a
robust characteristic length from mechanism topology is deliberately deferred
until this experiment confirms that the scaling formulation itself is useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from kimech._constraints import jacobian, residual
from kimech.joints import PrismaticJoint, RevoluteJoint
from kimech.solver import _RESIDUAL_TOL

from scale_robustness import (
    CASE_BUILDERS,
    SCALES,
    StudyCase,
    _pack_guess,
    _residual_components,
    _safe_condition,
    run_case,
)


@dataclass(frozen=True)
class ScaledStep:
    index: int
    input_value: float
    scipy_success: bool
    accepted: bool
    scaled_residual_inf: float
    normalized_linear_residual_inf: float
    angular_residual_inf: float
    scaled_jacobian_condition: float
    nfev: int
    njev: int
    message: str


@dataclass(frozen=True)
class Comparison:
    case: str
    scale: float
    raw_status: str
    scaled_status: str
    raw_solved: int
    scaled_solved: int
    total: int
    raw_fail_index: int | None
    scaled_fail_index: int | None
    raw_max_dimensionless_residual: float
    scaled_max_residual: float
    raw_max_condition: float
    scaled_max_condition: float
    raw_max_nfev: int
    scaled_max_nfev: int


def _coordinate_scale(case: StudyCase) -> np.ndarray:
    """Return diagonal entries of D_q for q = D_q q_hat."""
    block = np.array([case.characteristic_length, case.characteristic_length, 1.0])
    return np.tile(block, len(case.mechanism.links))


def _residual_scale(case: StudyCase) -> np.ndarray:
    """Return diagonal entries of D_phi for Phi_hat = D_phi^-1 Phi."""
    length = case.characteristic_length
    entries: list[float] = []
    for joint in case.mechanism.joints:
        if isinstance(joint, RevoluteJoint):
            entries.extend((length, length))
        elif isinstance(joint, PrismaticJoint):
            entries.extend((length, 1.0))
        else:  # pragma: no cover - study guard for future joint types
            raise TypeError(f"unsupported joint type: {type(joint).__name__}")

    if isinstance(case.input_joint, RevoluteJoint):
        entries.append(1.0)
    elif isinstance(case.input_joint, PrismaticJoint):
        entries.append(length)
    else:  # pragma: no cover - study guard for future joint types
        raise TypeError(f"unsupported input joint type: {type(case.input_joint).__name__}")

    result = np.asarray(entries, dtype=float)
    if np.any(result <= 0.0) or not np.all(np.isfinite(result)):
        raise ValueError("residual scale must contain only positive finite values")
    return result


def _solve_scaled_step(
    case: StudyCase,
    index: int,
    input_value: float,
    initial_q_hat: np.ndarray,
    q_scale: np.ndarray,
    phi_scale: np.ndarray,
) -> tuple[ScaledStep, np.ndarray | None]:
    mechanism = case.mechanism
    links = mechanism.links
    joints = mechanism.joints

    def unpack(q_hat: np.ndarray) -> np.ndarray:
        return q_scale * q_hat

    def fun(q_hat: np.ndarray) -> np.ndarray:
        q = unpack(q_hat)
        phi = residual(mechanism, links, joints, case.input_joint, q, input_value)
        return phi / phi_scale

    def jac(q_hat: np.ndarray) -> np.ndarray:
        q = unpack(q_hat)
        matrix = jacobian(mechanism, links, joints, case.input_joint, q, input_value)
        return (matrix * q_scale[np.newaxis, :]) / phi_scale[:, np.newaxis]

    result = optimize.root(fun, initial_q_hat, jac=jac, method="hybr")
    try:
        candidate_hat = np.asarray(result.x, dtype=float)
    except (TypeError, ValueError):
        candidate_hat = np.empty(0, dtype=float)

    valid = (
        candidate_hat.shape == initial_q_hat.shape
        and np.all(np.isfinite(candidate_hat))
    )
    if not valid:
        record = ScaledStep(
            index=index,
            input_value=input_value,
            scipy_success=bool(result.success),
            accepted=False,
            scaled_residual_inf=float("nan"),
            normalized_linear_residual_inf=float("nan"),
            angular_residual_inf=float("nan"),
            scaled_jacobian_condition=float("nan"),
            nfev=int(getattr(result, "nfev", -1)),
            njev=int(getattr(result, "njev", -1)),
            message=str(result.message),
        )
        return record, None

    phi_hat = fun(candidate_hat)
    scaled_residual_inf = float(np.linalg.norm(phi_hat, ord=np.inf))
    scaled_condition = _safe_condition(jac(candidate_hat))

    candidate_q = unpack(candidate_hat)
    phi = residual(
        mechanism,
        links,
        joints,
        case.input_joint,
        candidate_q,
        input_value,
    )
    linear_inf, angular_inf = _residual_components(phi, joints, case.input_joint)
    normalized_linear_inf = linear_inf / case.characteristic_length

    accepted = (
        result.success is True
        and np.isfinite(scaled_residual_inf)
        and scaled_residual_inf <= _RESIDUAL_TOL
    )
    record = ScaledStep(
        index=index,
        input_value=input_value,
        scipy_success=bool(result.success),
        accepted=bool(accepted),
        scaled_residual_inf=scaled_residual_inf,
        normalized_linear_residual_inf=normalized_linear_inf,
        angular_residual_inf=angular_inf,
        scaled_jacobian_condition=scaled_condition,
        nfev=int(getattr(result, "nfev", -1)),
        njev=int(getattr(result, "njev", -1)),
        message=str(result.message),
    )
    return record, candidate_hat.copy()


def run_scaled_case(case: StudyCase) -> list[ScaledStep]:
    q_scale = _coordinate_scale(case)
    phi_scale = _residual_scale(case)
    current_q_hat = _pack_guess(case) / q_scale
    records: list[ScaledStep] = []

    for index, raw_value in enumerate(case.values):
        value = float(raw_value)
        record, candidate_hat = _solve_scaled_step(
            case,
            index,
            value,
            current_q_hat,
            q_scale,
            phi_scale,
        )
        records.append(record)
        if not record.accepted or candidate_hat is None:
            break
        current_q_hat = candidate_hat

    return records


def _finite_max(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite, default=float("nan"))


def _dimensionless_raw_residual(record) -> float:
    return max(record.normalized_linear_residual_inf, record.angular_residual_inf)


def compare_case(scale: float, builder) -> Comparison:
    raw_summary, raw_steps = run_case(scale, builder)
    case = builder(scale)
    scaled_steps = run_scaled_case(case)

    raw_complete = len(raw_steps) == len(case.values) and raw_steps[-1].accepted
    scaled_complete = (
        len(scaled_steps) == len(case.values) and scaled_steps[-1].accepted
    )

    return Comparison(
        case=case.name,
        scale=scale,
        raw_status="PASS" if raw_complete else "FAIL",
        scaled_status="PASS" if scaled_complete else "FAIL",
        raw_solved=sum(step.accepted for step in raw_steps),
        scaled_solved=sum(step.accepted for step in scaled_steps),
        total=len(case.values),
        raw_fail_index=raw_summary.failed_index,
        scaled_fail_index=None if scaled_complete else scaled_steps[-1].index,
        raw_max_dimensionless_residual=_finite_max(
            _dimensionless_raw_residual(step) for step in raw_steps
        ),
        scaled_max_residual=_finite_max(
            step.scaled_residual_inf for step in scaled_steps
        ),
        raw_max_condition=_finite_max(step.jacobian_condition for step in raw_steps),
        scaled_max_condition=_finite_max(
            step.scaled_jacobian_condition for step in scaled_steps
        ),
        raw_max_nfev=max((step.nfev for step in raw_steps), default=-1),
        scaled_max_nfev=max((step.nfev for step in scaled_steps), default=-1),
    )


def _optional_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def print_comparisons(records: list[Comparison]) -> None:
    headers = (
        "case",
        "scale",
        "raw",
        "scaled",
        "raw_n",
        "scaled_n",
        "raw_fail",
        "scaled_fail",
        "raw_dim_res",
        "scaled_res",
        "raw_cond",
        "scaled_cond",
        "raw_nfev",
        "scaled_nfev",
    )
    rows: list[tuple[str, ...]] = []
    for item in records:
        rows.append(
            (
                item.case,
                f"{item.scale:.0e}",
                item.raw_status,
                item.scaled_status,
                f"{item.raw_solved}/{item.total}",
                f"{item.scaled_solved}/{item.total}",
                _optional_int(item.raw_fail_index),
                _optional_int(item.scaled_fail_index),
                f"{item.raw_max_dimensionless_residual:.3e}",
                f"{item.scaled_max_residual:.3e}",
                f"{item.raw_max_condition:.3e}",
                f"{item.scaled_max_condition:.3e}",
                str(item.raw_max_nfev),
                str(item.scaled_max_nfev),
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(render(headers))
    print(render(tuple("-" * width for width in widths)))
    for row in rows:
        print(render(row))


def main() -> None:
    comparisons = [
        compare_case(scale, builder)
        for builder in CASE_BUILDERS
        for scale in SCALES
    ]

    print("Dimensionless prototype")
    print(f"Acceptance tolerance on scaled residual: {_RESIDUAL_TOL:.1e}\n")
    print_comparisons(comparisons)


if __name__ == "__main__":
    main()
