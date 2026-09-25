"""Study branch continuity and direction consistency on complex mechanisms.

For each mechanism, solve one full input revolution forward and then solve the
same requested samples in reverse order, using the forward endpoint as the
reverse initial guess.  Compare corresponding physical configurations after
reversing the second solution.

Run from the repository root with the development/viz extras installed::

    python playground/branch_continuity_study.py
    python playground/branch_continuity_study.py --csv branch-continuity.csv

This is an exploratory study, not a gating test.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from archimedes_trammel import build_mechanism as build_trammel
from klann import build_mechanism as build_klann
from theo_jansen import build_mechanism as build_theo_jansen
from watt_six_bar import build_mechanism as build_watt
from whitworth import build_mechanism as build_whitworth
from kimech import KinematicSolution, KinematicDriver, solve
from kimech._scaling import build_numerical_scaling


SAMPLE_COUNT = 181


@dataclass(frozen=True)
class ContinuityRecord:
    name: str
    samples: int
    links: int
    joints: int
    max_forward_reverse_pose_error: float
    max_periodic_closure_error: float
    max_forward_step: float
    max_reverse_step: float
    forward_subdivisions: int
    reverse_subdivisions: int
    max_condition_number: float
    min_singular_value: float


def _unpack_builder_result(builder):
    result = builder()
    mechanism = result[0]
    input_joint = result[1]
    initial_guess = result[-2]
    theta0 = float(result[-1])
    return mechanism, input_joint, initial_guess, theta0


def _wrapped_angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _pose_error(
    poses_a: np.ndarray,
    poses_b: np.ndarray,
    *,
    characteristic_length: float,
) -> np.ndarray:
    translation = np.linalg.norm(poses_a[:, :2] - poses_b[:, :2], axis=1)
    angle = np.abs(_wrapped_angle_difference(poses_a[:, 2], poses_b[:, 2]))
    return np.maximum(translation / characteristic_length, angle)


def _max_solution_difference(
    first: KinematicSolution,
    second: KinematicSolution,
    *,
    characteristic_length: float,
) -> float:
    maximum = 0.0
    for link in first.mechanism.links:
        errors = _pose_error(
            first.body_poses(link),
            second.body_poses(link),
            characteristic_length=characteristic_length,
        )
        maximum = max(maximum, float(np.max(errors)))
    return maximum


def _max_step(
    solution: KinematicSolution,
    *,
    characteristic_length: float,
) -> float:
    maximum = 0.0
    for link in solution.mechanism.links:
        poses = solution.body_poses(link)
        if len(poses) < 2:
            continue
        errors = _pose_error(
            poses[1:],
            poses[:-1],
            characteristic_length=characteristic_length,
        )
        maximum = max(maximum, float(np.max(errors)))
    return maximum


def _periodic_closure(
    solution: KinematicSolution,
    *,
    characteristic_length: float,
) -> float:
    maximum = 0.0
    for link in solution.mechanism.links:
        poses = solution.body_poses(link)
        errors = _pose_error(
            poses[-1:],
            poses[:1],
            characteristic_length=characteristic_length,
        )
        maximum = max(maximum, float(errors[0]))
    return maximum


def run_case(name: str, builder) -> ContinuityRecord:
    mechanism, input_joint, initial_guess, theta0 = _unpack_builder_result(builder)
    values = np.linspace(theta0, theta0 + 2.0 * np.pi, SAMPLE_COUNT)

    scaling = build_numerical_scaling(
        mechanism,
        mechanism.links,
        mechanism.joints,
        input_joint,
        values,
    )

    forward = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values,
        ),
        initial_guess=initial_guess,
    )
    reverse = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values[::-1],
        ),
        initial_guess=forward[-1],
    )
    reverse_aligned = reverse[::-1]

    diagnostics_forward = forward.diagnostics
    diagnostics_reverse = reverse.diagnostics
    if diagnostics_forward is None or diagnostics_reverse is None:
        raise RuntimeError("solve() did not return diagnostics")

    subdivision_forward = diagnostics_forward.subdivision_counts
    subdivision_reverse = diagnostics_reverse.subdivision_counts
    if subdivision_forward is None or subdivision_reverse is None:
        raise RuntimeError("adaptive subdivision diagnostics are unavailable")

    return ContinuityRecord(
        name=name,
        samples=len(values),
        links=len(mechanism.links),
        joints=len(mechanism.joints),
        max_forward_reverse_pose_error=_max_solution_difference(
            forward,
            reverse_aligned,
            characteristic_length=scaling.characteristic_length,
        ),
        max_periodic_closure_error=_periodic_closure(
            forward,
            characteristic_length=scaling.characteristic_length,
        ),
        max_forward_step=_max_step(
            forward,
            characteristic_length=scaling.characteristic_length,
        ),
        max_reverse_step=_max_step(
            reverse,
            characteristic_length=scaling.characteristic_length,
        ),
        forward_subdivisions=int(np.sum(subdivision_forward)),
        reverse_subdivisions=int(np.sum(subdivision_reverse)),
        max_condition_number=float(
            max(
                np.max(diagnostics_forward.condition_numbers),
                np.max(diagnostics_reverse.condition_numbers),
            )
        ),
        min_singular_value=float(
            min(
                np.min(diagnostics_forward.min_singular_values),
                np.min(diagnostics_reverse.min_singular_values),
            )
        ),
    )


CASES = (
    ("archimedes_trammel", build_trammel),
    ("whitworth", build_whitworth),
    ("watt_ii_six_bar", build_watt),
    ("klann", build_klann),
    ("theo_jansen", build_theo_jansen),
)


def _print_table(records: list[ContinuityRecord]) -> None:
    headers = (
        "case",
        "samples",
        "fwd/rev err",
        "closure err",
        "max fwd step",
        "max rev step",
        "fwd subdiv",
        "rev subdiv",
        "max cond",
        "min sigma",
    )
    rows = [
        (
            item.name,
            str(item.samples),
            f"{item.max_forward_reverse_pose_error:.3e}",
            f"{item.max_periodic_closure_error:.3e}",
            f"{item.max_forward_step:.3e}",
            f"{item.max_reverse_step:.3e}",
            str(item.forward_subdivisions),
            str(item.reverse_subdivisions),
            f"{item.max_condition_number:.3e}",
            f"{item.min_singular_value:.3e}",
        )
        for item in records
    ]

    widths = [len(value) for value in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row):
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(render(headers))
    print(render(tuple("-" * width for width in widths)))
    for row in rows:
        print(render(row))


def _write_csv(path: Path, records: list[ContinuityRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="write summary records to CSV")
    args = parser.parse_args()

    records = [run_case(name, builder) for name, builder in CASES]
    _print_table(records)

    if args.csv is not None:
        _write_csv(args.csv, records)
        print(f"\nWrote {len(records)} records to {args.csv}")


if __name__ == "__main__":
    main()
