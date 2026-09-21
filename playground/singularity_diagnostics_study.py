"""Study driven-solve singularities using Kimech's public diagnostics.

The study compares the same physical configurations under different prescribed
joint coordinates.  Its purpose is to distinguish:

1. regular passage through a geometric dead-center when the chosen input remains
   a valid local parameter;
2. a fold/toggle of the driven solve when the chosen input reaches an extremum;
3. near-singular conditioning before numerical rank is actually lost.

Run from an editable installation of the repository::

    python playground/singularity_diagnostics_study.py
    python playground/singularity_diagnostics_study.py --csv singularities.csv

This is an exploratory study, not a gating test.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kimech import Configuration, KinematicSolveError, Mechanism, solve
from kimech.joints import PrismaticJoint, RevoluteJoint

ANGLE_OFFSETS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.0)


@dataclass(frozen=True)
class DiagnosticRecord:
    mechanism: str
    configuration: str
    driver: str
    angle_offset: float
    reference_angle: float
    input_position: float
    condition_number: float
    min_singular_value: float
    rank: int


def _slider_crank():
    mechanism = Mechanism("slider_crank_singularity_study")
    origin = mechanism.ground.add_point("O", (0.0, 0.0))
    guide = mechanism.ground.add_point("G", (0.0, 0.0))

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    rod = mechanism.add_link("connecting_rod")
    rod_b = rod.add_point("B", (0.0, 0.0))
    rod_c = rod.add_point("C", (0.24, 0.0))

    slider = mechanism.add_link("slider")
    slider_c = slider.add_point("C", (0.0, 0.0))
    slider_guide = slider.add_point("G", (0.0, 0.0))

    crank_joint = mechanism.revolute(origin, crank_o, name="crank_input")
    mechanism.revolute(crank_b, rod_b)
    mechanism.revolute(rod_c, slider_c)
    slider_joint = mechanism.prismatic(
        guide,
        slider_guide,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="slider_input",
    )

    guess = {
        crank: (0.0, 0.0, 0.7),
        rod: (0.06, 0.05, -0.2),
        slider: (0.30, 0.0, 0.0),
    }
    return mechanism, crank_joint, slider_joint, guess


def _four_bar():
    mechanism = Mechanism("four_bar_singularity_study")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22, 0.0))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    crank_joint = mechanism.revolute(ground_a, crank_a, name="crank_input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    rocker_joint = mechanism.revolute(rocker_d, ground_d, name="rocker_input")

    guess = {
        crank: (0.0, 0.0, 0.8),
        coupler: (0.05, 0.06, 0.2),
        rocker: (0.30, 0.0, 2.2),
    }
    return mechanism, crank_joint, rocker_joint, guess


def _reference_configuration(
    mechanism: Mechanism,
    input_joint: RevoluteJoint,
    guess,
    target: float,
    *,
    start: float,
) -> Configuration:
    sample_count = max(2, int(abs(target - start) / 0.03) + 2)
    values = np.linspace(start, target, sample_count)
    return solve(
        mechanism,
        input_joint=input_joint,
        input_position=values,
        initial_guess=guess,
    )[-1]


def _record(
    mechanism_name: str,
    configuration_name: str,
    driver_name: str,
    angle_offset: float,
    reference_angle: float,
    solution,
) -> DiagnosticRecord:
    diagnostics = solution.diagnostics
    if diagnostics is None:
        raise RuntimeError("solve() did not return diagnostics")
    return DiagnosticRecord(
        mechanism=mechanism_name,
        configuration=configuration_name,
        driver=driver_name,
        angle_offset=angle_offset,
        reference_angle=reference_angle,
        input_position=float(solution.input_positions[0]),
        condition_number=float(diagnostics.condition_numbers[0]),
        min_singular_value=float(diagnostics.min_singular_values[0]),
        rank=int(diagnostics.ranks[0]),
    )


def _compare_drivers(
    *,
    mechanism: Mechanism,
    primary_joint: RevoluteJoint,
    alternate_joint: RevoluteJoint | PrismaticJoint,
    guess,
    configuration_name: str,
    singular_angle: float,
    approach_sign: float,
    start_angle: float,
) -> list[DiagnosticRecord]:
    records: list[DiagnosticRecord] = []

    for offset in ANGLE_OFFSETS:
        reference_angle = singular_angle + approach_sign * offset
        config = _reference_configuration(
            mechanism,
            primary_joint,
            guess,
            reference_angle,
            start=start_angle,
        )

        primary = solve(
            mechanism,
            input_joint=primary_joint,
            input_position=reference_angle,
            initial_guess=config,
        )
        records.append(
            _record(
                mechanism.name or "mechanism",
                configuration_name,
                primary_joint.name or "primary",
                offset,
                reference_angle,
                primary,
            )
        )

        alternate_value = config.joint_coordinate(alternate_joint)
        alternate = solve(
            mechanism,
            input_joint=alternate_joint,
            input_position=alternate_value,
            initial_guess=config,
        )
        records.append(
            _record(
                mechanism.name or "mechanism",
                configuration_name,
                alternate_joint.name or "alternate",
                offset,
                reference_angle,
                alternate,
            )
        )

    return records


def _lower_circle_intersection_angle(radius_a: float, radius_d: float, ground: float) -> float:
    """Return the polar angle of the lower intersection of two circles."""
    x = (radius_a**2 - radius_d**2 + ground**2) / (2.0 * ground)
    y_sq = radius_a**2 - x**2
    if y_sq < 0.0 and abs(y_sq) < 1e-15:
        y_sq = 0.0
    if y_sq < 0.0:
        raise ValueError("circles do not intersect")
    return float(np.arctan2(-np.sqrt(y_sq), x))


def slider_crank_records() -> list[DiagnosticRecord]:
    mechanism, crank_joint, slider_joint, guess = _slider_crank()
    records: list[DiagnosticRecord] = []

    records.extend(
        _compare_drivers(
            mechanism=mechanism,
            primary_joint=crank_joint,
            alternate_joint=slider_joint,
            guess=guess,
            configuration_name="right_dead_center",
            singular_angle=0.0,
            approach_sign=1.0,
            start_angle=0.7,
        )
    )
    records.extend(
        _compare_drivers(
            mechanism=mechanism,
            primary_joint=crank_joint,
            alternate_joint=slider_joint,
            guess=guess,
            configuration_name="left_dead_center",
            singular_angle=np.pi,
            approach_sign=-1.0,
            start_angle=0.7,
        )
    )
    return records


def four_bar_records() -> list[DiagnosticRecord]:
    mechanism, crank_joint, rocker_joint, guess = _four_bar()

    crank = 0.08
    coupler = 0.22
    rocker = 0.18
    ground = 0.30

    extended_axis = _lower_circle_intersection_angle(
        crank + coupler,
        rocker,
        ground,
    )
    extended_toggle = float(extended_axis % (2.0 * np.pi))

    folded_axis = _lower_circle_intersection_angle(
        coupler - crank,
        rocker,
        ground,
    )
    folded_toggle = float((folded_axis + np.pi) % (2.0 * np.pi))

    records: list[DiagnosticRecord] = []
    records.extend(
        _compare_drivers(
            mechanism=mechanism,
            primary_joint=crank_joint,
            alternate_joint=rocker_joint,
            guess=guess,
            configuration_name="folded_toggle",
            singular_angle=folded_toggle,
            approach_sign=-1.0,
            start_angle=0.8,
        )
    )
    records.extend(
        _compare_drivers(
            mechanism=mechanism,
            primary_joint=crank_joint,
            alternate_joint=rocker_joint,
            guess=guess,
            configuration_name="extended_toggle",
            singular_angle=extended_toggle,
            approach_sign=-1.0,
            start_angle=0.8,
        )
    )
    return records


def _probe_beyond_limit() -> None:
    mechanism, crank_joint, slider_joint, guess = _slider_crank()
    dead_center = _reference_configuration(
        mechanism,
        crank_joint,
        guess,
        0.0,
        start=0.7,
    )
    slider_limit = dead_center.joint_coordinate(slider_joint)

    try:
        solve(
            mechanism,
            input_joint=slider_joint,
            input_position=slider_limit + 1e-6,
            initial_guess=dead_center,
        )
    except KinematicSolveError as error:
        print("\nBeyond-limit probe: expected failure")
        print(f"  requested slider coordinate: {slider_limit + 1e-6:.9f}")
        print(f"  physical limit:             {slider_limit:.9f}")
        print(f"  solver message: {error}")
    else:
        print("\nBeyond-limit probe unexpectedly found a solution.")


def _print_table(records: list[DiagnosticRecord]) -> None:
    headers = (
        "mechanism",
        "configuration",
        "driver",
        "|dtheta|",
        "cond(Jhat)",
        "sigma_min",
        "rank",
    )
    rows = [
        (
            item.mechanism,
            item.configuration,
            item.driver,
            f"{item.angle_offset:.0e}",
            f"{item.condition_number:.6e}",
            f"{item.min_singular_value:.6e}",
            str(item.rank),
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


def _write_csv(path: Path, records: list[DiagnosticRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="write all diagnostic records to CSV")
    args = parser.parse_args()

    records = slider_crank_records() + four_bar_records()
    _print_table(records)
    _probe_beyond_limit()

    if args.csv is not None:
        _write_csv(args.csv, records)
        print(f"\nWrote {len(records)} records to {args.csv}")


if __name__ == "__main__":
    main()
