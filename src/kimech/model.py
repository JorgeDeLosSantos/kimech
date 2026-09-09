"""Declarative domain model for planar mechanisms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._geometry import as_vector2

if TYPE_CHECKING:
    from .joints import PrismaticJoint, RevoluteJoint


@dataclass(frozen=True, slots=True)
class Point:
    """A named point rigidly attached to a link or to ground."""

    name: str
    body: Link | Ground
    _coordinates: tuple[float, float]

    @property
    def local(self) -> np.ndarray:
        """Return a copy of the point coordinates in the owning body frame."""
        return np.asarray(self._coordinates, dtype=float).copy()


class Link:
    """A mobile planar rigid body with an arbitrary local reference frame."""

    def __init__(self, mechanism: Mechanism, name: str) -> None:
        self._mechanism = mechanism
        self._name = name
        self._points: list[Point] = []
        self._points_by_name: dict[str, Point] = {}

    @property
    def mechanism(self) -> Mechanism:
        return self._mechanism

    @property
    def name(self) -> str:
        return self._name

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(self._points)

    def add_point(self, name: str, coordinates: Sequence[float]) -> Point:
        point_name = _validate_name(name, entity="point")
        if point_name in self._points_by_name:
            raise ValueError(f"point name {point_name!r} already exists on link {self.name!r}")
        point = Point(point_name, self, as_vector2(coordinates, name="point coordinates"))
        self._points.append(point)
        self._points_by_name[point_name] = point
        return point

    def __getitem__(self, name: str) -> Point:
        return self._points_by_name[name]

    def __repr__(self) -> str:
        return f"Link(name={self.name!r})"


class Ground:
    """The unique fixed body defining a mechanism's global frame."""

    def __init__(self, mechanism: Mechanism) -> None:
        self._mechanism = mechanism
        self._points: list[Point] = []
        self._points_by_name: dict[str, Point] = {}

    @property
    def mechanism(self) -> Mechanism:
        return self._mechanism

    @property
    def name(self) -> str:
        return "ground"

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(self._points)

    def add_point(self, name: str, coordinates: Sequence[float]) -> Point:
        point_name = _validate_name(name, entity="point")
        if point_name in self._points_by_name:
            raise ValueError(f"point name {point_name!r} already exists on ground")
        point = Point(point_name, self, as_vector2(coordinates, name="point coordinates"))
        self._points.append(point)
        self._points_by_name[point_name] = point
        return point

    def __getitem__(self, name: str) -> Point:
        return self._points_by_name[name]

    def __repr__(self) -> str:
        return "Ground()"


class Mechanism:
    """Declarative model of a planar rigid-body mechanism."""

    def __init__(self, name: str | None = None) -> None:
        self._name = None if name is None else _validate_name(name, entity="mechanism")
        self._ground = Ground(self)
        self._links: list[Link] = []
        self._links_by_name: dict[str, Link] = {}
        self._joints: list[RevoluteJoint | PrismaticJoint] = []

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def ground(self) -> Ground:
        return self._ground

    @property
    def links(self) -> tuple[Link, ...]:
        return tuple(self._links)

    @property
    def joints(self) -> tuple[RevoluteJoint | PrismaticJoint, ...]:
        return tuple(self._joints)

    def add_link(self, name: str) -> Link:
        link_name = _validate_name(name, entity="link")
        if link_name == self.ground.name:
            raise ValueError("'ground' is reserved for the fixed body")
        if link_name in self._links_by_name:
            raise ValueError(f"link name {link_name!r} already exists")
        link = Link(self, link_name)
        self._links.append(link)
        self._links_by_name[link_name] = link
        return link

    def revolute(
        self,
        point_a: Point,
        point_b: Point,
        *,
        name: str | None = None,
    ) -> RevoluteJoint:
        from .joints import RevoluteJoint

        joint = RevoluteJoint(point_a, point_b, name=name)
        self._ensure_joint_belongs_here(joint)
        self._joints.append(joint)
        return joint

    def prismatic(
        self,
        point_a: Point,
        point_b: Point,
        *,
        axis_a: Sequence[float],
        axis_b: Sequence[float],
        name: str | None = None,
    ) -> PrismaticJoint:
        from .joints import PrismaticJoint

        joint = PrismaticJoint(point_a, point_b, axis_a=axis_a, axis_b=axis_b, name=name)
        self._ensure_joint_belongs_here(joint)
        self._joints.append(joint)
        return joint

    def __getitem__(self, name: str) -> Link:
        return self._links_by_name[name]

    def _ensure_joint_belongs_here(self, joint: RevoluteJoint | PrismaticJoint) -> None:
        if joint.point_a.body.mechanism is not self or joint.point_b.body.mechanism is not self:
            raise ValueError("joint points must belong to this mechanism")

    def __repr__(self) -> str:
        return f"Mechanism(name={self.name!r}, links={len(self.links)}, joints={len(self.joints)})"


def _validate_name(value: str, *, entity: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{entity} name must be a string")
    name = value.strip()
    if not name:
        raise ValueError(f"{entity} name must not be empty")
    return name
