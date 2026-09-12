"""Solver-independent kinematic result objects."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np

from ._geometry import perpendicular, rotation_matrix
from .joints import PrismaticJoint, RevoluteJoint
from .model import Ground, Link, Mechanism, Point

_Joint = RevoluteJoint | PrismaticJoint
_Body = Link | Ground


class Configuration:
    """One valid kinematic configuration of a planar mechanism."""

    __slots__ = (
        "_coordinate_accelerations",
        "_coordinate_velocities",
        "_coordinates",
        "_input_acceleration",
        "_input_joint",
        "_input_value",
        "_input_velocity",
        "_links",
        "_mechanism",
    )

    def __init__(
        self,
        mechanism: Mechanism,
        coordinates: Sequence[float] | np.ndarray,
        *,
        coordinate_velocities: Sequence[float] | np.ndarray | None = None,
        coordinate_accelerations: Sequence[float] | np.ndarray | None = None,
        input_joint: _Joint | None = None,
        input_value: float | None = None,
        input_velocity: float | None = None,
        input_acceleration: float | None = None,
    ) -> None:
        _validate_mechanism(mechanism)
        self._initialize(
            mechanism,
            mechanism.links,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            input_joint=input_joint,
            input_value=input_value,
            input_velocity=input_velocity,
            input_acceleration=input_acceleration,
        )

    @classmethod
    def _from_snapshot(
        cls,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        coordinates: Sequence[float] | np.ndarray,
        *,
        coordinate_velocities: Sequence[float] | np.ndarray | None = None,
        coordinate_accelerations: Sequence[float] | np.ndarray | None = None,
        input_joint: _Joint | None = None,
        input_value: float | None = None,
        input_velocity: float | None = None,
        input_acceleration: float | None = None,
    ) -> Configuration:
        configuration = cls.__new__(cls)
        configuration._initialize(
            mechanism,
            links,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            input_joint=input_joint,
            input_value=input_value,
            input_velocity=input_velocity,
            input_acceleration=input_acceleration,
        )
        return configuration

    def _initialize(
        self,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        coordinates: Sequence[float] | np.ndarray,
        *,
        coordinate_velocities: Sequence[float] | np.ndarray | None,
        coordinate_accelerations: Sequence[float] | np.ndarray | None,
        input_joint: _Joint | None,
        input_value: float | None,
        input_velocity: float | None,
        input_acceleration: float | None,
    ) -> None:
        if input_joint is not None:
            _validate_joint(mechanism, links, input_joint)
        if coordinate_accelerations is not None and coordinate_velocities is None:
            raise ValueError("coordinate_accelerations requires coordinate_velocities")
        if input_acceleration is not None and input_velocity is None:
            raise ValueError("input_acceleration requires input_velocity")

        state_shape = (3 * len(links),)
        self._mechanism = mechanism
        self._links = links
        self._coordinates = _finite_float_array(
            coordinates,
            name="coordinates",
            shape=state_shape,
        )
        self._coordinate_velocities = _optional_finite_float_array(
            coordinate_velocities,
            name="coordinate_velocities",
            shape=state_shape,
        )
        self._coordinate_accelerations = _optional_finite_float_array(
            coordinate_accelerations,
            name="coordinate_accelerations",
            shape=state_shape,
        )
        self._input_joint = input_joint
        self._input_value = _optional_finite_scalar(input_value, name="input_value")
        self._input_velocity = _optional_finite_scalar(input_velocity, name="input_velocity")
        self._input_acceleration = _optional_finite_scalar(
            input_acceleration,
            name="input_acceleration",
        )

    @property
    def mechanism(self) -> Mechanism:
        return self._mechanism

    @property
    def input_joint(self) -> _Joint | None:
        return self._input_joint

    @property
    def input_value(self) -> float | None:
        return self._input_value

    @property
    def input_velocity(self) -> float | None:
        return self._input_velocity

    @property
    def input_acceleration(self) -> float | None:
        return self._input_acceleration

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates.copy()

    @property
    def coordinate_velocities(self) -> np.ndarray:
        if self._coordinate_velocities is None:
            raise ValueError("velocity data is not available in this configuration")
        return self._coordinate_velocities.copy()

    @property
    def coordinate_accelerations(self) -> np.ndarray:
        if self._coordinate_accelerations is None:
            raise ValueError("acceleration data is not available in this configuration")
        return self._coordinate_accelerations.copy()

    @property
    def has_velocity(self) -> bool:
        return self._coordinate_velocities is not None

    @property
    def has_acceleration(self) -> bool:
        return self._coordinate_accelerations is not None

    def body_pose(self, body: _Body) -> np.ndarray:
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros(3, dtype=float)
        start = 3 * index
        return self._coordinates[start : start + 3].copy()

    def body_velocity(self, body: _Body) -> np.ndarray:
        index = _body_index(self._mechanism, self._links, body)
        velocities = self._require_velocity()
        if index is None:
            return np.zeros(3, dtype=float)
        start = 3 * index
        return velocities[start : start + 3].copy()

    def body_acceleration(self, body: _Body) -> np.ndarray:
        index = _body_index(self._mechanism, self._links, body)
        accelerations = self._require_acceleration()
        if index is None:
            return np.zeros(3, dtype=float)
        start = 3 * index
        return accelerations[start : start + 3].copy()

    def position(self, point: Point) -> np.ndarray:
        _validate_point(self._mechanism, self._links, point)
        if point.body is self._mechanism.ground:
            return point.local
        x, y, theta = self.body_pose(point.body)
        return np.array([x, y], dtype=float) + rotation_matrix(theta) @ point.local

    def velocity(self, point: Point) -> np.ndarray:
        _validate_point(self._mechanism, self._links, point)
        body_velocity = self.body_velocity(point.body)
        if point.body is self._mechanism.ground:
            return np.zeros(2, dtype=float)
        theta = self.body_pose(point.body)[2]
        rotational = rotation_matrix(theta) @ perpendicular(point.local)
        return body_velocity[:2] + body_velocity[2] * rotational

    def acceleration(self, point: Point) -> np.ndarray:
        _validate_point(self._mechanism, self._links, point)
        body_acceleration = self.body_acceleration(point.body)
        body_velocity = self.body_velocity(point.body)
        if point.body is self._mechanism.ground:
            return np.zeros(2, dtype=float)
        theta = self.body_pose(point.body)[2]
        rotation = rotation_matrix(theta)
        local = np.asarray(point.local, dtype=float)
        tangent = rotation @ perpendicular(local)
        radial = rotation @ local
        return (
            body_acceleration[:2]
            + body_acceleration[2] * tangent
            - body_velocity[2] ** 2 * radial
        )

    def joint_coordinate(self, joint: _Joint) -> float:
        _validate_joint(self._mechanism, self._links, joint)
        if isinstance(joint, RevoluteJoint):
            theta_a = self.body_pose(joint.point_a.body)[2]
            theta_b = self.body_pose(joint.point_b.body)[2]
            return float(theta_b - theta_a)
        pose_a = self.body_pose(joint.point_a.body)
        axis_a = rotation_matrix(pose_a[2]) @ np.asarray(joint.axis_a, dtype=float)
        displacement = self.position(joint.point_b) - self.position(joint.point_a)
        return float(axis_a @ displacement)

    def joint_velocity(self, joint: _Joint) -> float:
        _validate_joint(self._mechanism, self._links, joint)
        if isinstance(joint, RevoluteJoint):
            omega_a = self.body_velocity(joint.point_a.body)[2]
            omega_b = self.body_velocity(joint.point_b.body)[2]
            return float(omega_b - omega_a)
        pose_a = self.body_pose(joint.point_a.body)
        axis_a = rotation_matrix(pose_a[2]) @ np.asarray(joint.axis_a, dtype=float)
        relative_velocity = self.velocity(joint.point_b) - self.velocity(joint.point_a)
        return float(axis_a @ relative_velocity)

    def joint_acceleration(self, joint: _Joint) -> float:
        _validate_joint(self._mechanism, self._links, joint)
        if isinstance(joint, RevoluteJoint):
            alpha_a = self.body_acceleration(joint.point_a.body)[2]
            alpha_b = self.body_acceleration(joint.point_b.body)[2]
            return float(alpha_b - alpha_a)
        pose_a = self.body_pose(joint.point_a.body)
        axis_a = rotation_matrix(pose_a[2]) @ np.asarray(joint.axis_a, dtype=float)
        relative_acceleration = self.acceleration(joint.point_b) - self.acceleration(joint.point_a)
        coordinate = self.joint_coordinate(joint)
        omega_a = self.body_velocity(joint.point_a.body)[2]
        return float(axis_a @ relative_acceleration + coordinate * omega_a**2)

    def _require_velocity(self) -> np.ndarray:
        if self._coordinate_velocities is None:
            raise ValueError("velocity data is not available in this configuration")
        return self._coordinate_velocities

    def _require_acceleration(self) -> np.ndarray:
        if self._coordinate_accelerations is None:
            raise ValueError("acceleration data is not available in this configuration")
        return self._coordinate_accelerations


class KinematicSolution:
    """An ordered sequence of accepted mechanism configurations."""

    __slots__ = (
        "_coordinate_accelerations",
        "_coordinate_velocities",
        "_coordinates",
        "_input_accelerations",
        "_input_joint",
        "_input_values",
        "_input_velocities",
        "_links",
        "_mechanism",
    )

    def __init__(
        self,
        mechanism: Mechanism,
        input_joint: _Joint,
        input_values: Sequence[float] | np.ndarray,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        *,
        coordinate_velocities: Sequence[Sequence[float]] | np.ndarray | None = None,
        coordinate_accelerations: Sequence[Sequence[float]] | np.ndarray | None = None,
        input_velocities: Sequence[float] | np.ndarray | None = None,
        input_accelerations: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        _validate_mechanism(mechanism)
        self._initialize(
            mechanism,
            mechanism.links,
            input_joint,
            input_values,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            input_velocities=input_velocities,
            input_accelerations=input_accelerations,
        )

    @classmethod
    def _from_snapshot(
        cls,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        input_joint: _Joint,
        input_values: Sequence[float] | np.ndarray,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        *,
        coordinate_velocities: Sequence[Sequence[float]] | np.ndarray | None = None,
        coordinate_accelerations: Sequence[Sequence[float]] | np.ndarray | None = None,
        input_velocities: Sequence[float] | np.ndarray | None = None,
        input_accelerations: Sequence[float] | np.ndarray | None = None,
    ) -> KinematicSolution:
        solution = cls.__new__(cls)
        solution._initialize(
            mechanism,
            links,
            input_joint,
            input_values,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            input_velocities=input_velocities,
            input_accelerations=input_accelerations,
        )
        return solution

    def _initialize(
        self,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        input_joint: _Joint,
        input_values: Sequence[float] | np.ndarray,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        *,
        coordinate_velocities: Sequence[Sequence[float]] | np.ndarray | None,
        coordinate_accelerations: Sequence[Sequence[float]] | np.ndarray | None,
        input_velocities: Sequence[float] | np.ndarray | None,
        input_accelerations: Sequence[float] | np.ndarray | None,
    ) -> None:
        _validate_mechanism(mechanism)
        _validate_joint(mechanism, links, input_joint)
        if coordinate_accelerations is not None and coordinate_velocities is None:
            raise ValueError("coordinate_accelerations requires coordinate_velocities")
        if input_accelerations is not None and input_velocities is None:
            raise ValueError("input_accelerations requires input_velocities")

        values = _finite_float_array(input_values, name="input_values", ndim=1)
        state_shape = (len(values), 3 * len(links))
        coordinate_array = _finite_float_array(
            coordinates,
            name="coordinates",
            shape=state_shape,
        )
        velocity_array = _optional_finite_float_array(
            coordinate_velocities,
            name="coordinate_velocities",
            shape=state_shape,
        )
        acceleration_array = _optional_finite_float_array(
            coordinate_accelerations,
            name="coordinate_accelerations",
            shape=state_shape,
        )
        input_shape = (len(values),)
        input_velocity_array = _optional_finite_float_array(
            input_velocities,
            name="input_velocities",
            shape=input_shape,
        )
        input_acceleration_array = _optional_finite_float_array(
            input_accelerations,
            name="input_accelerations",
            shape=input_shape,
        )

        self._mechanism = mechanism
        self._links = links
        self._input_joint = input_joint
        self._input_values = values
        self._input_velocities = input_velocity_array
        self._input_accelerations = input_acceleration_array
        self._coordinates = coordinate_array
        self._coordinate_velocities = velocity_array
        self._coordinate_accelerations = acceleration_array

    @property
    def mechanism(self) -> Mechanism:
        return self._mechanism

    @property
    def input_joint(self) -> _Joint:
        return self._input_joint

    @property
    def input_values(self) -> np.ndarray:
        return self._input_values.copy()

    @property
    def input_velocities(self) -> np.ndarray | None:
        if self._input_velocities is None:
            return None
        return self._input_velocities.copy()

    @property
    def input_accelerations(self) -> np.ndarray | None:
        if self._input_accelerations is None:
            return None
        return self._input_accelerations.copy()

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates.copy()

    @property
    def coordinate_velocities(self) -> np.ndarray:
        if self._coordinate_velocities is None:
            raise ValueError("velocity data is not available in this solution")
        return self._coordinate_velocities.copy()

    @property
    def coordinate_accelerations(self) -> np.ndarray:
        if self._coordinate_accelerations is None:
            raise ValueError("acceleration data is not available in this solution")
        return self._coordinate_accelerations.copy()

    @property
    def has_velocity(self) -> bool:
        return self._coordinate_velocities is not None

    @property
    def has_acceleration(self) -> bool:
        return self._coordinate_accelerations is not None

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
            coordinate_velocities=(
                None if self._coordinate_velocities is None else self._coordinate_velocities[item]
            ),
            coordinate_accelerations=(
                None
                if self._coordinate_accelerations is None
                else self._coordinate_accelerations[item]
            ),
            input_joint=self._input_joint,
            input_value=self._input_values[item],
            input_velocity=(
                None if self._input_velocities is None else self._input_velocities[item]
            ),
            input_acceleration=(
                None
                if self._input_accelerations is None
                else self._input_accelerations[item]
            ),
        )

    def point_path(self, point: Point) -> np.ndarray:
        _validate_point(self._mechanism, self._links, point)
        path = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            path[index] = self[index].position(point)
        return path

    def point_velocities(self, point: Point) -> np.ndarray:
        _validate_point(self._mechanism, self._links, point)
        self._require_velocity()
        values = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].velocity(point)
        return values

    def point_accelerations(self, point: Point) -> np.ndarray:
        _validate_point(self._mechanism, self._links, point)
        self._require_acceleration()
        values = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].acceleration(point)
        return values

    def body_poses(self, body: _Body) -> np.ndarray:
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return self._coordinates[:, start : start + 3].copy()

    def body_velocities(self, body: _Body) -> np.ndarray:
        index = _body_index(self._mechanism, self._links, body)
        velocities = self._require_velocity()
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return velocities[:, start : start + 3].copy()

    def body_accelerations(self, body: _Body) -> np.ndarray:
        index = _body_index(self._mechanism, self._links, body)
        accelerations = self._require_acceleration()
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return accelerations[:, start : start + 3].copy()

    def joint_coordinates(self, joint: _Joint) -> np.ndarray:
        _validate_joint(self._mechanism, self._links, joint)
        values = np.empty(len(self), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].joint_coordinate(joint)
        return values

    def joint_velocities(self, joint: _Joint) -> np.ndarray:
        _validate_joint(self._mechanism, self._links, joint)
        self._require_velocity()
        values = np.empty(len(self), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].joint_velocity(joint)
        return values

    def joint_accelerations(self, joint: _Joint) -> np.ndarray:
        _validate_joint(self._mechanism, self._links, joint)
        self._require_acceleration()
        values = np.empty(len(self), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].joint_acceleration(joint)
        return values

    def _require_velocity(self) -> np.ndarray:
        if self._coordinate_velocities is None:
            raise ValueError("velocity data is not available in this solution")
        return self._coordinate_velocities

    def _require_acceleration(self) -> np.ndarray:
        if self._coordinate_accelerations is None:
            raise ValueError("acceleration data is not available in this solution")
        return self._coordinate_accelerations


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


def _optional_finite_float_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if value is None:
        return None
    return _finite_float_array(value, name=name, shape=shape)


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