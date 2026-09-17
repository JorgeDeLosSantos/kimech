"""Characterize position-solver sensitivity to geometric scale.

This study intentionally uses Kimech's current position formulation without
changing solver tolerances or scaling.  It calls the same residual/Jacobian and
SciPy ``hybr`` method used by :func:`kimech.solve`, then records enough detail to
distinguish three broad failure modes:

1. SciPy converges but Kimech's absolute residual acceptance rejects the result.
2. The nonlinear solver itself behaves differently as geometric scale changes.
3. Sequential continuation loses a solution that can still be recovered from
   the original initial guess.

Run from an environment where the repository is installed, for example::

    python playground/scale_robustness.py
    python playground/scale_robustness.py --csv scale-robustness.csv

No result from this file is a gating test.  The purpose is to collect evidence
before changing the 0.3.0 solver.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from scipy import optimize

from kimech import Mechanism
from kimech._constraints import jacobian, residual
from kimech.joints import PrismaticJoint, RevoluteJoint
from kimech.model import Link
from kimech.solver import _RESIDUAL_TOL


Joint: TypeAlias = RevoluteJoint | PrismaticJoint
SCALES = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)


@dataclass(frozen=True)
class StudyCase:
    """One mechanism family at one geometric scale."""

    name: str
    mechanism: Mechanism
    input_joint: Joint
    values: np.ndarray
    initial_guess: dict[Link, tuple[float, float, float]]
    characteristic_length: float


@dataclass(frozen=True)
class StepRecord:
    """Diagnostics for one nonlinear position solve."""

    case: str
    scale: float
    index: int
    input_value: float
    scipy_success: bool
    accepted: bool
    residual_inf: float
    linear_residual_inf: float
    normalized_linear_residual_inf: float
    angular_residual_inf: float
    jacobian_condition: float
    nfev: int
    njev: int
    translation_step_over_length: float
    angular_step_inf: float
    message: str


@dataclass(frozen=True)
class SummaryRecord:
    """One-line summary for a complete scale experiment."""

    case: str
    scale: float
    solved: int
    total: int
    failed_index: int | None
    failed_input: float | None
    scipy_success: bool
    accepted: bool
    residual_inf: float
    normalized_linear_residual_inf: float
    angular_residual_inf: float
    jacobian_condition: float
    nfev: int
    retry_from_initial_accepted: bool | None


def _scaled_pose(
    pose: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    x, y, theta = pose
    return (scale * x, scale * y, theta)


def _control_case(scale: float) -> StudyCase:
    """A single driven revolute joint with nonzero translational coordinates."""
    mechanism = Mechanism("scale_control")
    fixed = mechanism.ground.add_point("O", (120.0 * scale, -40.0 * scale))

    link = mechanism.add_link("link")
    pivot = link.add_point("O", (30.0 * scale, 20.0 * scale))
    link.add_point("P", (130.0 * scale, 20.0 * scale))

    input_joint = mechanism.revolute(fixed, pivot, name="input")
    guess = {link: _scaled_pose((90.0, -60.0, 0.0), scale)}
    return StudyCase(
        name="control_revolute",
        mechanism=mechanism,
        input_joint=input_joint,
        values=np.linspace(0.0, 2.0 * np.pi, 361),
        initial_guess=guess,
        characteristic_length=100.0 * scale,
    )


def _baseline_four_bar(scale: float) -> StudyCase:
    """Four-bar already used by Kimech's differential acceptance tests."""
    mechanism = Mechanism("baseline_four_bar")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30 * scale, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08 * scale, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22 * scale, 0.0))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18 * scale, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)

    guess = {
        crank: _scaled_pose((0.0, 0.0, 0.8), scale),
        coupler: _scaled_pose((0.05, 0.06, 0.2), scale),
        rocker: _scaled_pose((0.30, 0.0, 2.2), scale),
    }
    return StudyCase(
        name="baseline_four_bar",
        mechanism=mechanism,
        input_joint=input_joint,
        values=np.linspace(0.8, 1.3, 25),
        initial_guess=guess,
        characteristic_length=0.30 * scale,
    )


def _problem_four_bar(scale: float) -> StudyCase:
    """Reproduce the mm-scale sweep failure observed before 0.3.0.

    ``scale=1`` is the original model expressed numerically in millimetres.
    ``scale=1e-3`` is the same geometry after dividing every linear quantity by
    1000; that scaled model was observed to complete the sweep successfully.
    """
    mechanism = Mechanism("problem_four_bar")
    ground_o = mechanism.ground.add_point("O", (0.0, 0.0))
    ground_c = mechanism.ground.add_point("C", (120.0 * scale, 0.0))

    link_2 = mechanism.add_link("link_2")
    link_2_o = link_2.add_point("O", (-25.0 * scale, 0.0))
    link_2_a = link_2.add_point("A", (25.0 * scale, 0.0))

    link_3 = mechanism.add_link("link_3")
    link_3_a = link_3.add_point("A", (0.0, 0.0))
    link_3_b = link_3.add_point("B", (200.0 * scale, 0.0))

    link_4 = mechanism.add_link("link_4")
    link_4_b = link_4.add_point("B", (200.0 * scale, 0.0))
    link_4_c = link_4.add_point("C", (0.0, 0.0))

    input_joint = mechanism.revolute(ground_o, link_2_o, name="input")
    mechanism.revolute(link_2_a, link_3_a)
    mechanism.revolute(link_3_b, link_4_b)
    mechanism.revolute(link_4_c, ground_c)

    guess = {
        link_2: _scaled_pose((25.0, 0.0, 0.0), scale),
        link_3: _scaled_pose((50.0, 0.0, 1.4), scale),
        link_4: _scaled_pose((120.0, 0.0, 1.75), scale),
    }
    return StudyCase(
        name="problem_four_bar",
        mechanism=mechanism,
        input_joint=input_joint,
        values=np.linspace(0.0, 2.0 * np.pi, 361),
        initial_guess=guess,
        characteristic_length=200.0 * scale,
    )


CASE_BUILDERS = (_control_case, _baseline_four_bar, _problem_four_bar)


def _pack_guess(case: StudyCase) -> np.ndarray:
    packed = np.empty(3 * len(case.mechanism.links), dtype=float)
    for index, link in enumerate(case.mechanism.links):
        packed[3 * index : 3 * index + 3] = case.initial_guess[link]
    return packed


def _residual_components(
    phi: np.ndarray,
    joints: tuple[Joint, ...],
    input_joint: Joint,
) -> tuple[float, float]:
    """Return separate infinity norms for linear and angular equations."""
    linear: list[float] = []
    angular: list[float] = []

    for index, joint in enumerate(joints):
        row = phi[2 * index : 2 * index + 2]
        if isinstance(joint, RevoluteJoint):
            linear.extend(float(value) for value in row)
        else:
            linear.append(float(row[0]))
            angular.append(float(row[1]))

    if isinstance(input_joint, RevoluteJoint):
        angular.append(float(phi[-1]))
    else:
        linear.append(float(phi[-1]))

    linear_inf = max((abs(value) for value in linear), default=0.0)
    angular_inf = max((abs(value) for value in angular), default=0.0)
    return linear_inf, angular_inf


def _safe_condition(matrix: np.ndarray) -> float:
    try:
        value = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def _solve_step(
    case: StudyCase,
    scale: float,
    index: int,
    input_value: float,
    initial_q: np.ndarray,
) -> tuple[StepRecord, np.ndarray | None]:
    mechanism = case.mechanism
    links = mechanism.links
    joints = mechanism.joints

    def fun(q: np.ndarray) -> np.ndarray:
        return residual(mechanism, links, joints, case.input_joint, q, input_value)

    def jac(q: np.ndarray) -> np.ndarray:
        return jacobian(mechanism, links, joints, case.input_joint, q, input_value)

    result = optimize.root(fun, initial_q, jac=jac, method="hybr")
    expected_shape = initial_q.shape
    try:
        candidate = np.asarray(result.x, dtype=float)
    except (TypeError, ValueError):
        candidate = np.empty(0, dtype=float)

    candidate_valid = (
        candidate.shape == expected_shape and np.all(np.isfinite(candidate))
    )
    if not candidate_valid:
        record = StepRecord(
            case=case.name,
            scale=scale,
            index=index,
            input_value=input_value,
            scipy_success=bool(result.success),
            accepted=False,
            residual_inf=float("nan"),
            linear_residual_inf=float("nan"),
            normalized_linear_residual_inf=float("nan"),
            angular_residual_inf=float("nan"),
            jacobian_condition=float("nan"),
            nfev=int(getattr(result, "nfev", -1)),
            njev=int(getattr(result, "njev", -1)),
            translation_step_over_length=float("nan"),
            angular_step_inf=float("nan"),
            message=str(result.message),
        )
        return record, None

    phi = fun(candidate)
    residual_inf = float(np.linalg.norm(phi, ord=np.inf))
    linear_inf, angular_inf = _residual_components(phi, joints, case.input_joint)
    normalized_linear_inf = linear_inf / case.characteristic_length
    condition = _safe_condition(jac(candidate))

    delta = candidate - initial_q
    translations = delta.reshape(-1, 3)[:, :2]
    angles = delta.reshape(-1, 3)[:, 2]
    translation_step = float(
        np.max(np.linalg.norm(translations, axis=1)) / case.characteristic_length
    )
    angular_step = float(np.max(np.abs(angles)))

    accepted = (
        result.success is True
        and np.isfinite(residual_inf)
        and residual_inf <= _RESIDUAL_TOL
    )
    record = StepRecord(
        case=case.name,
        scale=scale,
        index=index,
        input_value=input_value,
        scipy_success=bool(result.success),
        accepted=bool(accepted),
        residual_inf=residual_inf,
        linear_residual_inf=linear_inf,
        normalized_linear_residual_inf=normalized_linear_inf,
        angular_residual_inf=angular_inf,
        jacobian_condition=condition,
        nfev=int(getattr(result, "nfev", -1)),
        njev=int(getattr(result, "njev", -1)),
        translation_step_over_length=translation_step,
        angular_step_inf=angular_step,
        message=str(result.message),
    )
    return record, candidate.copy()


def _summarize(
    case: StudyCase,
    scale: float,
    records: list[StepRecord],
    retry_from_initial_accepted: bool | None,
) -> SummaryRecord:
    last = records[-1]
    complete = len(records) == len(case.values) and last.accepted
    if complete:
        representative = max(records, key=lambda item: item.residual_inf)
        failed_index = None
        failed_input = None
    else:
        representative = last
        failed_index = last.index
        failed_input = last.input_value

    return SummaryRecord(
        case=case.name,
        scale=scale,
        solved=sum(record.accepted for record in records),
        total=len(case.values),
        failed_index=failed_index,
        failed_input=failed_input,
        scipy_success=representative.scipy_success,
        accepted=complete,
        residual_inf=representative.residual_inf,
        normalized_linear_residual_inf=representative.normalized_linear_residual_inf,
        angular_residual_inf=representative.angular_residual_inf,
        jacobian_condition=representative.jacobian_condition,
        nfev=representative.nfev,
        retry_from_initial_accepted=retry_from_initial_accepted,
    )


def run_case(scale: float, builder) -> tuple[SummaryRecord, list[StepRecord]]:
    case = builder(scale)
    initial_q = _pack_guess(case)
    current_q = initial_q.copy()
    records: list[StepRecord] = []
    retry_from_initial_accepted: bool | None = None

    for index, raw_value in enumerate(case.values):
        value = float(raw_value)
        record, candidate = _solve_step(case, scale, index, value, current_q)
        records.append(record)
        if not record.accepted or candidate is None:
            if index > 0:
                retry, _ = _solve_step(case, scale, index, value, initial_q)
                retry_from_initial_accepted = retry.accepted
            break
        current_q = candidate

    return _summarize(case, scale, records, retry_from_initial_accepted), records


def _format_optional_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _format_optional_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def print_summary(records: list[SummaryRecord]) -> None:
    headers = (
        "case",
        "scale",
        "status",
        "solved",
        "fail_i",
        "input",
        "scipy",
        "res_inf",
        "lin/L",
        "ang_inf",
        "cond(J)",
        "nfev",
        "retry",
    )
    rows: list[tuple[str, ...]] = []
    for item in records:
        retry = (
            "-"
            if item.retry_from_initial_accepted is None
            else ("yes" if item.retry_from_initial_accepted else "no")
        )
        rows.append(
            (
                item.case,
                f"{item.scale:.0e}",
                "PASS" if item.accepted else "FAIL",
                f"{item.solved}/{item.total}",
                _format_optional_int(item.failed_index),
                _format_optional_float(item.failed_input),
                "yes" if item.scipy_success else "no",
                f"{item.residual_inf:.3e}",
                f"{item.normalized_linear_residual_inf:.3e}",
                f"{item.angular_residual_inf:.3e}",
                f"{item.jacobian_condition:.3e}",
                str(item.nfev),
                retry,
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


def write_csv(path: Path, records: list[StepRecord]) -> None:
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="write every nonlinear solve to this CSV file",
    )
    args = parser.parse_args()

    summaries: list[SummaryRecord] = []
    steps: list[StepRecord] = []
    for builder in CASE_BUILDERS:
        for scale in SCALES:
            summary, records = run_case(scale, builder)
            summaries.append(summary)
            steps.extend(records)

    print(f"Kimech residual acceptance tolerance: {_RESIDUAL_TOL:.1e}\n")
    print_summary(summaries)

    if args.csv is not None:
        write_csv(args.csv, steps)
        print(f"\nWrote {len(steps)} step records to {args.csv}")


if __name__ == "__main__":
    main()
