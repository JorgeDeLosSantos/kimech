"""Public joint domain objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ._geometry import normalize_axis
from .model import Point


@dataclass(frozen=True, slots=True)
class RevoluteJoint:
    """Planar revolute joint joining two points on distinct bodies."""

    point_a: Point
    point_b: Point
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_joint_points(self.point_a, self.point_b)
        object.__setattr__(self, "name", _normalize_optional_name(self.name))


@dataclass(frozen=True, slots=True)
class PrismaticJoint:
    """Planar prismatic joint defined by local reference points and axes."""

    point_a: Point
    point_b: Point
    axis_a: Sequence[float] = field(repr=False)
    axis_b: Sequence[float] = field(repr=False)
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_joint_points(self.point_a, self.point_b)
        object.__setattr__(self, "axis_a", normalize_axis(self.axis_a, name="axis_a"))
        object.__setattr__(self, "axis_b", normalize_axis(self.axis_b, name="axis_b"))
        object.__setattr__(self, "name", _normalize_optional_name(self.name))

    @property
    def axis_a_array(self) -> np.ndarray:
        return np.asarray(self.axis_a, dtype=float).copy()

    @property
    def axis_b_array(self) -> np.ndarray:
        return np.asarray(self.axis_b, dtype=float).copy()


def _validate_joint_points(point_a: Point, point_b: Point) -> None:
    if not isinstance(point_a, Point) or not isinstance(point_b, Point):
        raise TypeError("joint endpoints must be Point objects")
    if point_a.body is point_b.body:
        raise ValueError("a joint must connect two distinct bodies")
    if point_a.body.mechanism is not point_b.body.mechanism:
        raise ValueError("joint points must belong to the same mechanism")


def _normalize_optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("joint name must be a string or None")
    name = value.strip()
    if not name:
        raise ValueError("joint name must not be empty")
    return name
