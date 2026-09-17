"""Private numerical scaling helpers for kinematic solves."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .joints import PrismaticJoint, RevoluteJoint
from .model import Link, Mechanism, Point

_Joint = RevoluteJoint | PrismaticJoint


@dataclass(frozen=True, slots=True)
class NumericalScaling:
    """Diagonal scaling for generalized coordinates and constraint equations."""

    characteristic_length: float
    coordinate_scale: np.ndarray
    residual_scale: np.ndarray

    def scale_coordinates(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float) / self.coordinate_scale

    def unscale_coordinates(self, q_hat: np.ndarray) -> np.ndarray:
        return self.coordinate_scale * np.asarray(q_hat, dtype=float)

    def scale_residual(self, phi: np.ndarray) -> np.ndarray:
        return np.asarray(phi, dtype=float) / self.residual_scale

    def scale_jacobian(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        return (
            values * self.coordinate_scale[np.newaxis, :]
        ) / self.residual_scale[:, np.newaxis]

    def scale_rhs(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(rhs, dtype=float) / self.residual_scale

    def unscale_state(self, state_hat: np.ndarray) -> np.ndarray:
        return self.coordinate_scale * np.asarray(state_hat, dtype=float)


def build_numerical_scaling(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    input_values: np.ndarray,
) -> NumericalScaling:
    """Build one global dimensionless scaling for a complete solve call."""
    length = infer_characteristic_length(
        mechanism,
        links,
        joints,
        input_joint,
        input_values,
    )
    coordinate_block = np.array([length, length, 1.0], dtype=float)
    coordinate_scale = np.tile(coordinate_block, len(links))

    residual_entries: list[float] = []
    for joint in joints:
        if isinstance(joint, RevoluteJoint):
            residual_entries.extend((length, length))
        elif isinstance(joint, PrismaticJoint):
            residual_entries.extend((length, 1.0))
        else:  # pragma: no cover - guarded by the public model
            raise TypeError(f"unsupported joint type: {type(joint).__name__}")

    if isinstance(input_joint, RevoluteJoint):
        residual_entries.append(1.0)
    elif isinstance(input_joint, PrismaticJoint):
        residual_entries.append(length)
    else:  # pragma: no cover - guarded by solve()
        raise TypeError(f"unsupported input joint type: {type(input_joint).__name__}")

    residual_scale = np.asarray(residual_entries, dtype=float)
    return NumericalScaling(length, coordinate_scale, residual_scale)


def infer_characteristic_length(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    input_values: np.ndarray,
) -> float:
    """Infer a positive linear scale from solve-relevant mechanism geometry."""
    grouped = _structural_points_by_body(joints)
    candidates: list[float] = []

    # Same-body structural spans are invariant to local/global frame origins.
    for points in grouped.values():
        for point_a, point_b in combinations(points, 2):
            candidates.append(float(np.linalg.norm(point_a.local - point_b.local)))

    # Mobile-body offsets are lever arms multiplying angular coordinates in J.
    for link in links:
        for point in grouped.get(link, ()):  # ground offsets are excluded intentionally
            candidates.append(float(np.linalg.norm(point.local)))

    # A pure prismatic mechanism can have no geometric span at all.  The
    # prescribed displacement history then supplies the only linear scale.
    if isinstance(input_joint, PrismaticJoint):
        values = np.asarray(input_values, dtype=float)
        if values.size:
            candidates.append(float(np.max(np.abs(values))))

    positive = [
        value
        for value in candidates
        if np.isfinite(value) and value > 0.0
    ]
    return max(positive, default=1.0)


def _structural_points_by_body(
    joints: tuple[_Joint, ...],
) -> dict[object, list[Point]]:
    grouped: dict[object, list[Point]] = {}
    seen: dict[object, set[int]] = {}
    for joint in joints:
        for point in (joint.point_a, joint.point_b):
            body = point.body
            body_seen = seen.setdefault(body, set())
            if id(point) in body_seen:
                continue
            grouped.setdefault(body, []).append(point)
            body_seen.add(id(point))
    return grouped
