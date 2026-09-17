"""Study automatic characteristic-length inference for Kimech 0.3.0.

This is the third numerical-robustness study.  It keeps ``kimech.solve``
untouched and asks how the characteristic length used by the dimensionless
prototype can be inferred from the solve problem itself.

Candidate policy
----------------
The inferred length is the largest positive finite value among:

* pairwise distances between structural points on the same body, including
  ground;
* distances from each mobile body's local origin to its structural points;
* absolute prescribed input positions when the input joint is prismatic.

Only points referenced by joints are structural.  Nonstructural points therefore
cannot change the solver scale.  Absolute ground-point coordinates are never
used, so translating the global frame cannot change the inferred length.

The mobile-body local offsets are intentional: unlike a global translation,
changing a mobile body's local frame changes the numerical lever arm multiplying
its angular coordinate in the Jacobian.  The scale should respond to that
parameterization even when the physical mechanism is unchanged.

If the problem contains no positive linear scale at all, the fallback is 1.0.
That value is dimensionless bookkeeping rather than a physical unit.

Run from the repository root::

    python playground/characteristic_length_study.py
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np

from kimech import Mechanism
from kimech.joints import PrismaticJoint

from scale_robustness import CASE_BUILDERS, SCALES, StudyCase
from scale_robustness_dimensionless import run_scaled_case


def _structural_points_by_body(mechanism: Mechanism) -> dict[object, list[object]]:
    """Return unique joint-referenced points grouped by their owning body."""
    grouped: dict[object, list[object]] = {}
    seen: dict[object, set[int]] = {}
    for joint in mechanism.joints:
        for point in (joint.point_a, joint.point_b):
            body = point.body
            body_seen = seen.setdefault(body, set())
            if id(point) in body_seen:
                continue
            grouped.setdefault(body, []).append(point)
            body_seen.add(id(point))
    return grouped


def infer_characteristic_length(case: StudyCase) -> float:
    """Infer one global positive characteristic length for a solve problem."""
    candidates: list[float] = []
    grouped = _structural_points_by_body(case.mechanism)

    # Intrinsic spans are invariant to the local/global frame origins.
    for points in grouped.values():
        for point_a, point_b in combinations(points, 2):
            candidates.append(float(np.linalg.norm(point_a.local - point_b.local)))

    # Mobile-body offsets are numerically relevant rotational lever arms.
    for link in case.mechanism.links:
        for point in grouped.get(link, ()):  # ground offsets are deliberately excluded
            candidates.append(float(np.linalg.norm(point.local)))

    # A pure slider may have no geometric span at all.  Its prescribed natural
    # coordinate is then the only linear scale visible to the solve problem.
    if isinstance(case.input_joint, PrismaticJoint):
        values = np.asarray(case.values, dtype=float)
        if values.size:
            candidates.append(float(np.max(np.abs(values))))

    positive = [
        value
        for value in candidates
        if np.isfinite(value) and value > 0.0
    ]
    return max(positive, default=1.0)


def _scaled_summary(case: StudyCase) -> tuple[bool, float, int]:
    steps = run_scaled_case(case)
    complete = len(steps) == len(case.values) and steps[-1].accepted
    max_condition = max(
        (step.scaled_jacobian_condition for step in steps if np.isfinite(step.scaled_jacobian_condition)),
        default=float("nan"),
    )
    max_nfev = max((step.nfev for step in steps), default=-1)
    return complete, max_condition, max_nfev


def standard_cases() -> None:
    print("Standard A2 cases with inferred characteristic length")
    headers = (
        "case",
        "scale",
        "manual_L",
        "inferred_L",
        "ratio",
        "status",
        "max_cond",
        "max_nfev",
    )
    rows: list[tuple[str, ...]] = []

    for builder in CASE_BUILDERS:
        for scale in SCALES:
            manual = builder(scale)
            inferred_length = infer_characteristic_length(manual)
            inferred = replace(manual, characteristic_length=inferred_length)
            complete, condition, nfev = _scaled_summary(inferred)
            rows.append(
                (
                    manual.name,
                    f"{scale:.0e}",
                    f"{manual.characteristic_length:.6g}",
                    f"{inferred_length:.6g}",
                    f"{inferred_length / manual.characteristic_length:.6g}",
                    "PASS" if complete else "FAIL",
                    f"{condition:.3e}",
                    str(nfev),
                )
            )

    _print_table(headers, rows)


def covariance_checks() -> None:
    print("\nScale covariance: inferred L / scale should be constant")
    headers = ("case", "min(L/s)", "max(L/s)", "relative_spread")
    rows: list[tuple[str, ...]] = []
    for builder in CASE_BUILDERS:
        normalized = np.asarray(
            [infer_characteristic_length(builder(scale)) / scale for scale in SCALES],
            dtype=float,
        )
        spread = float((normalized.max() - normalized.min()) / normalized.max())
        rows.append(
            (
                builder(1.0).name,
                f"{normalized.min():.12g}",
                f"{normalized.max():.12g}",
                f"{spread:.3e}",
            )
        )
    _print_table(headers, rows)


def _ground_probe(translation: tuple[float, float], *, add_poi: bool) -> StudyCase:
    tx, ty = translation
    mechanism = Mechanism("ground_translation_probe")
    ground_a = mechanism.ground.add_point("A", (tx, ty))
    ground_b = mechanism.ground.add_point("B", (tx + 3.0, ty + 4.0))

    link_a = mechanism.add_link("link_a")
    link_a_a = link_a.add_point("A", (0.0, 0.0))
    if add_poi:
        link_a.add_point("far_poi", (1.0e12, -1.0e12))

    link_b = mechanism.add_link("link_b")
    link_b_b = link_b.add_point("B", (0.0, 0.0))

    input_joint = mechanism.revolute(ground_a, link_a_a, name="input")
    mechanism.revolute(ground_b, link_b_b, name="support")
    return StudyCase(
        name="ground_translation_probe",
        mechanism=mechanism,
        input_joint=input_joint,
        values=np.array([0.0]),
        initial_guess={link_a: (tx, ty, 0.0), link_b: (tx + 3.0, ty + 4.0, 0.0)},
        characteristic_length=1.0,
    )


def frame_and_poi_checks() -> None:
    print("\nGround translation and nonstructural-point immunity")
    origin = _ground_probe((0.0, 0.0), add_poi=False)
    translated = _ground_probe((1.0e9, -2.0e9), add_poi=False)
    with_poi = _ground_probe((0.0, 0.0), add_poi=True)
    print(f"origin L      = {infer_characteristic_length(origin):.12g}")
    print(f"translated L  = {infer_characteristic_length(translated):.12g}")
    print(f"far-POI L     = {infer_characteristic_length(with_poi):.12g}")


def _single_revolute_offset(offset: float) -> StudyCase:
    mechanism = Mechanism("single_revolute_offset")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("O", (offset, 0.0))
    input_joint = mechanism.revolute(fixed, pivot, name="input")
    return StudyCase(
        name="single_revolute_offset",
        mechanism=mechanism,
        input_joint=input_joint,
        values=np.linspace(0.0, 2.0 * np.pi, 361),
        initial_guess={link: (-offset, 0.0, 0.0)},
        characteristic_length=1.0,
    )


def local_frame_stress() -> None:
    print("\nMobile-frame offset stress test")
    headers = ("offset", "inferred_L", "status", "max_cond", "max_nfev")
    rows: list[tuple[str, ...]] = []
    for offset in SCALES:
        raw = _single_revolute_offset(offset)
        length = infer_characteristic_length(raw)
        case = replace(raw, characteristic_length=length)
        complete, condition, nfev = _scaled_summary(case)
        rows.append(
            (
                f"{offset:.0e}",
                f"{length:.6g}",
                "PASS" if complete else "FAIL",
                f"{condition:.3e}",
                str(nfev),
            )
        )
    _print_table(headers, rows)


def _pure_prismatic(scale: float) -> StudyCase:
    mechanism = Mechanism("pure_prismatic")
    fixed = mechanism.ground.add_point("G", (0.0, 0.0))
    slider = mechanism.add_link("slider")
    moving = slider.add_point("G", (0.0, 0.0))
    input_joint = mechanism.prismatic(
        fixed,
        moving,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="input",
    )
    return StudyCase(
        name="pure_prismatic",
        mechanism=mechanism,
        input_joint=input_joint,
        values=scale * np.linspace(0.0, 100.0, 101),
        initial_guess={slider: (0.0, 0.0, 0.0)},
        characteristic_length=1.0,
    )


def prismatic_scale_check() -> None:
    print("\nDegenerate pure-prismatic input: prescribed displacement supplies L")
    headers = ("scale", "inferred_L", "L/scale", "status", "max_cond", "max_nfev")
    rows: list[tuple[str, ...]] = []
    for scale in SCALES:
        raw = _pure_prismatic(scale)
        length = infer_characteristic_length(raw)
        case = replace(raw, characteristic_length=length)
        complete, condition, nfev = _scaled_summary(case)
        rows.append(
            (
                f"{scale:.0e}",
                f"{length:.6g}",
                f"{length / scale:.6g}",
                "PASS" if complete else "FAIL",
                f"{condition:.3e}",
                str(nfev),
            )
        )
    _print_table(headers, rows)


def fallback_check() -> None:
    mechanism = Mechanism("zero_length_revolute")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    moving = link.add_point("O", (0.0, 0.0))
    input_joint = mechanism.revolute(fixed, moving, name="input")
    case = StudyCase(
        name="zero_length_revolute",
        mechanism=mechanism,
        input_joint=input_joint,
        values=np.array([0.5]),
        initial_guess={link: (0.0, 0.0, 0.0)},
        characteristic_length=1.0,
    )
    print("\nZero-linear-scale fallback")
    print(f"inferred L = {infer_characteristic_length(case):.12g}")


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
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
    standard_cases()
    covariance_checks()
    frame_and_poi_checks()
    local_frame_stress()
    prismatic_scale_check()
    fallback_check()


if __name__ == "__main__":
    main()
