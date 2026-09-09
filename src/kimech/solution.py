"""Solver-independent kinematic result objects."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np

from ._geometry import rotation_matrix
from .joints import PrismaticJoint, RevoluteJoint
from .model import Ground, Link, Mechanism, Point

_Joint = RevoluteJoint | PrismaticJoint
_Body = Link | Ground


class Configuration:
    """One valid position configuration of a planar mechanism."""

    __slots__ = ("_coordinates", "_input_joint", "_input_value", "_links", "_mechanism")

    def __init__(
        self,
        mechanism: Mechanism,
        coordinates: Sequence[float] | np.ndarray,
        *,
        input_joint: _Joint | None = None,
        input_value: float | None = None,
    ) -> None:
        _validate_mechanism(mechanism)
        self._initialize(
            mechanism,
            mechanism.links,
            coordinates,
            input_joint=input_joint,
            input_value=input_value,
        )

    @classmethod
    def _from_snapshot(
        cls,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        coordinates: Sequence[float] | np.ndarray,
        *,
        input_joint: _Joint | None = None,
        input_value: float | None = None,
    ) -> Configuration:
        configuration = cls.__new__(cls)
        configuration._initialize(
            mechanism,
            links,
            coordinates,
            input_joint=input_joint,
            input_value=input_value,
        )
        return configuration

    def _initialize(
        self,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        coordinates: Sequence[float] | np.ndarray,
        *,
        input_joint: _Joint | None,
        input_value: float | None,
    ) -> None:
        if input_joint is not None:
            _validate_joint(mechanism, links, input_joint)

        self._mechanism = mechanism
        self._links = links
        self._coordinates = _finite_float_array(
            coordinates,
            name="coordinates",
            shape=(3 * len(links),),
        )
        self._input_joint = input_joint
        self._input_value = _optional_finite_scalar(input_value, name="input_value")

    @property
    def mechanism(self) -> Mechanism:
        """Return the mechanism represented by this configuration."""
        return self._mechanism

    @property
    def input_joint(self) -> _Joint | None:
        """Return the prescribed joint associated with this configuration, if any."""
        return self._input_joint

    @property
    def input_value(self) -> float | None:
        """Return the prescribed joint value associated with this configuration."""
        return self._input_value

    @property
    def coordinates(self) -> np.ndarray:
        """Return a safe copy of the generalized coordinate vector."""
        return self._coordinates.copy()

    def pose(self, body: _Body) -> np.ndarray:
        """Return ``(x, y, theta)`` for a mobile link or the identity for ground."""
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros(3, dtype=float)
        start = 3 * index
        return self._coordinates[start : start + 3].copy()

    def position(self, point: Point) -> np.ndarray:
        """Return the global position of a point attached to a mechanism body."""
        _validate_point(self._mechanism, self._links, point)
        if point.body is self._mechanism.ground:
            return point.local

        x, y, theta = self.pose(point.body)
        return np.array([x, y], dtype=float) + rotation_matrix(theta) @ point.local

    def joint_coordinate(self, joint: _Joint) -> float:
        """Return the natural, unwrapped coordinate of a mechanism joint."""
        _validate_joint(self._mechanism, self._links, joint)

        if isinstance(joint, RevoluteJoint):
            theta_a = self.pose(joint.point_a.body)[2]
            theta_b = self.pose(joint.point_b.body)[2]
            return float(theta_b - theta_a)

        pose_a = self.pose(joint.point_a.body)
        axis_a = rotation_matrix(pose_a[2]) @ np.asarray(joint.axis_a, dtype=float)
        displacement = self.position(joint.point_b) - self.position(joint.point_a)
        return float(axis_a @ displacement)


class KinematicSolution:
    """An ordered sequence of accepted mechanism configurations."""

    __slots__ = ("_coordinates", "_input_joint", "_input_values", "_links", "_mechanism")

    def __init__(
        self,
        mechanism: Mechanism,
        input_joint: _Joint,
        input_values: Sequence[float] | np.ndarray,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
    ) -> None:
        _validate_mechanism(mechanism)
        self._initialize(
            mechanism,
            mechanism.links,
            input_joint,
            input_values,
            coordinates,
        )

    @classmethod
    def _from_snapshot(
        cls,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        input_joint: _Joint,
        input_values: Sequence[float] | np.ndarray,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
    ) -> KinematicSolution:
        solution = cls.__new__(cls)
        solution._initialize(
            mechanism,
            links,
            input_joint,
            input_values,
            coordinates,
        )
        return solution

    def _initialize(
        self,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        input_joint: _Joint,
        input_values: Sequence[float] | np.ndarray,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
    ) -> None:
        _validate_mechanism(mechanism)
        _validate_joint(mechanism, links, input_joint)

        values = _finite_float_array(input_values, name="input_values", ndim=1)
        coordinate_array = _finite_float_array(coordinates, name="coordinates", ndim=2)
        expected_shape = (len(values), 3 * len(links))
        if coordinate_array.shape != expected_shape:
            raise ValueError(f"coordinates must have shape {expected_shape}")

        self._mechanism = mechanism
        self._links = links
        self._input_joint = input_joint
        self._input_values = values
        self._coordinates = coordinate_array

    @property
    def mechanism(self) -> Mechanism:
        """Return the mechanism represented by this solution."""
        return self._mechanism

    @property
    def input_joint(self) -> _Joint:
        """Return the joint whose coordinate parameterizes the solution."""
        return self._input_joint

    @property
    def input_values(self) -> np.ndarray:
        """Return a safe copy of the ordered prescribed values."""
        return self._input_values.copy()

    @property
    def coordinates(self) -> np.ndarray:
        """Return a safe copy of the generalized coordinate matrix."""
        return self._coordinates.copy()

    def __len__(self) -> int:
        return len(self._input_values)

    def __getitem__(self, index: int) -> Configuration:
        if isinstance(index, slice):
            raise TypeError("KinematicSolution does not support slicing")
        item = operator.index(index)
        return Configuration._from_snapshot(
            self._mechanism,
            self._links,
            self._coordinates[item],
            input_joint=self._input_joint,
            input_value=self._input_values[item],
        )

    def point_path(self, point: Point) -> np.ndarray:
        """Return the global point positions for all configurations."""
        _validate_point(self._mechanism, self._links, point)
        path = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            path[index] = self[index].position(point)
        return path

    def link_poses(self, body: _Body) -> np.ndarray:
        """Return the pose history of a mobile link or ground."""
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return self._coordinates[:, start : start + 3].copy()

    def joint_coordinates(self, joint: _Joint) -> np.ndarray:
        """Return the natural joint coordinate for all configurations."""
        _validate_joint(self._mechanism, self._links, joint)
        values = np.empty(len(self), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].joint_coordinate(joint)
        return values


def _validate_mechanism(mechanism: object) -> None:
    if not isinstance(mechanism, Mechanism):
        raise TypeError("mechanism must be a Mechanism")


def _body_index(mechanism: Mechanism, links: tuple[Link, ...], body: object) -> int | None:
    if not isinstance(body, (Link, Ground)):
        raise TypeError("body must be a Link or Ground")
    if body is mechanism.ground:
        return None
    for index, link in enumerate(links):
        if body is link:
            return index
    raise ValueError("body does not belong to this mechanism")


def _validate_point(mechanism: Mechanism, links: tuple[Link, ...], point: object) -> None:
    if not isinstance(point, Point):
        raise TypeError("point must be a Point")
    try:
        _body_index(mechanism, links, point.body)
    except ValueError as error:
        raise ValueError("point does not belong to this mechanism") from error
    if not any(point is owned_point for owned_point in point.body.points):
        raise ValueError("point does not belong to this mechanism")


def _validate_joint(mechanism: Mechanism, links: tuple[Link, ...], joint: object) -> None:
    if not isinstance(joint, (RevoluteJoint, PrismaticJoint)):
        raise TypeError("joint must be a RevoluteJoint or PrismaticJoint")
    if not any(joint is owned_joint for owned_joint in mechanism.joints):
        raise ValueError("joint does not belong to this mechanism")
    try:
        _body_index(mechanism, links, joint.point_a.body)
        _body_index(mechanism, links, joint.point_b.body)
    except ValueError as error:
        raise ValueError("joint bodies do not belong to this result") from error


def _finite_float_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _optional_finite_scalar(value: object, *, name: str) -> float | None:
    if value is None:
        return None
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