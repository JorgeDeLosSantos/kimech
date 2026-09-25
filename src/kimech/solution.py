"""Solver-independent kinematic result objects."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence

import numpy as np

from ._geometry import perpendicular, rotation_matrix
from .diagnostics import SolveDiagnostics
from .driver import KinematicDriver
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
        "_driver",
        "_links",
        "_mechanism",
        "_time",
    )

    def __init__(
        self,
        mechanism: Mechanism,
        coordinates: Sequence[float] | np.ndarray,
        *,
        coordinate_velocities: Sequence[float] | np.ndarray | None = None,
        coordinate_accelerations: Sequence[float] | np.ndarray | None = None,
        driver: KinematicDriver | None = None,
        time: float | None = None,
    ) -> None:
        _validate_mechanism(mechanism)
        self._initialize(
            mechanism,
            mechanism.links,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            driver=driver,
            time=time,
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
        driver: KinematicDriver | None = None,
        time: float | None = None,
    ) -> Configuration:
        configuration = cls.__new__(cls)
        configuration._initialize(
            mechanism,
            links,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            driver=driver,
            time=time,
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
        driver: KinematicDriver | None,
        time: float | None,
    ) -> None:
        if driver is not None:
            if not isinstance(driver, KinematicDriver):
                raise TypeError("driver must be a KinematicDriver or None")
            _validate_joint(mechanism, links, driver.joint)
            if driver.sample_count != 1:
                raise ValueError("configuration driver must contain exactly one sample")
        if coordinate_accelerations is not None and coordinate_velocities is None:
            raise ValueError("coordinate_accelerations requires coordinate_velocities")

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
        self._driver = driver
        self._time = _optional_finite_scalar(time, name="time")

    @property
    def mechanism(self) -> Mechanism:
        """Return the mechanism represented by this configuration."""
        return self._mechanism

    @property
    def driver(self) -> KinematicDriver | None:
        """Return the prescribed kinematic driver for this configuration."""
        return self._driver

    @property
    def time(self) -> float | None:
        """Return the physical sample time associated with this configuration."""
        return self._time

    @property
    def coordinates(self) -> np.ndarray:
        """Return a safe copy of the generalized coordinate vector."""
        return self._coordinates.copy()

    @property
    def coordinate_velocities(self) -> np.ndarray:
        """Return a safe copy of the generalized velocity vector."""
        if self._coordinate_velocities is None:
            raise ValueError("velocity data is not available in this configuration")
        return self._coordinate_velocities.copy()

    @property
    def coordinate_accelerations(self) -> np.ndarray:
        """Return a safe copy of the generalized acceleration vector."""
        if self._coordinate_accelerations is None:
            raise ValueError("acceleration data is not available in this configuration")
        return self._coordinate_accelerations.copy()

    @property
    def has_velocity(self) -> bool:
        """Return whether generalized velocity state is available."""
        return self._coordinate_velocities is not None

    @property
    def has_acceleration(self) -> bool:
        """Return whether generalized acceleration state is available."""
        return self._coordinate_accelerations is not None

    def body_pose(self, body: _Body) -> np.ndarray:
        """Return ``(x, y, theta)`` for a mobile link or the identity for ground."""
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros(3, dtype=float)
        start = 3 * index
        return self._coordinates[start : start + 3].copy()

    def body_velocity(self, body: _Body) -> np.ndarray:
        """Return ``(vx, vy, omega)`` for a mechanism body."""
        index = _body_index(self._mechanism, self._links, body)
        velocities = self._require_velocity()
        if index is None:
            return np.zeros(3, dtype=float)
        start = 3 * index
        return velocities[start : start + 3].copy()

    def body_acceleration(self, body: _Body) -> np.ndarray:
        """Return ``(ax, ay, alpha)`` for a mechanism body."""
        index = _body_index(self._mechanism, self._links, body)
        accelerations = self._require_acceleration()
        if index is None:
            return np.zeros(3, dtype=float)
        start = 3 * index
        return accelerations[start : start + 3].copy()

    def point_position(self, point: Point) -> np.ndarray:
        """Return the global position of a point attached to a mechanism body."""
        _validate_point(self._mechanism, self._links, point)
        if point.body is self._mechanism.ground:
            return point.local
        x, y, theta = self.body_pose(point.body)
        return np.array([x, y], dtype=float) + rotation_matrix(theta) @ point.local

    def point_velocity(self, point: Point) -> np.ndarray:
        """Return the global velocity of a point attached to a mechanism body."""
        _validate_point(self._mechanism, self._links, point)
        body_velocity = self.body_velocity(point.body)
        if point.body is self._mechanism.ground:
            return np.zeros(2, dtype=float)
        theta = self.body_pose(point.body)[2]
        rotational = rotation_matrix(theta) @ perpendicular(point.local)
        return body_velocity[:2] + body_velocity[2] * rotational

    def point_acceleration(self, point: Point) -> np.ndarray:
        """Return the global acceleration of a point attached to a mechanism body."""
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
        """Return the natural, unwrapped coordinate of a mechanism joint."""
        _validate_joint(self._mechanism, self._links, joint)
        if isinstance(joint, RevoluteJoint):
            theta_a = self.body_pose(joint.point_a.body)[2]
            theta_b = self.body_pose(joint.point_b.body)[2]
            return float(theta_b - theta_a)
        pose_a = self.body_pose(joint.point_a.body)
        axis_a = rotation_matrix(pose_a[2]) @ np.asarray(joint.axis_a, dtype=float)
        displacement = self.point_position(joint.point_b) - self.point_position(joint.point_a)
        return float(axis_a @ displacement)

    def joint_velocity(self, joint: _Joint) -> float:
        """Return the time derivative of a joint's natural coordinate."""
        _validate_joint(self._mechanism, self._links, joint)
        if isinstance(joint, RevoluteJoint):
            omega_a = self.body_velocity(joint.point_a.body)[2]
            omega_b = self.body_velocity(joint.point_b.body)[2]
            return float(omega_b - omega_a)
        pose_a = self.body_pose(joint.point_a.body)
        axis_a = rotation_matrix(pose_a[2]) @ np.asarray(joint.axis_a, dtype=float)
        relative_velocity = self.point_velocity(joint.point_b) - self.point_velocity(joint.point_a)
        return float(axis_a @ relative_velocity)

    def joint_acceleration(self, joint: _Joint) -> float:
        """Return the second time derivative of a joint's natural coordinate."""
        _validate_joint(self._mechanism, self._links, joint)
        if isinstance(joint, RevoluteJoint):
            alpha_a = self.body_acceleration(joint.point_a.body)[2]
            alpha_b = self.body_acceleration(joint.point_b.body)[2]
            return float(alpha_b - alpha_a)
        pose_a = self.body_pose(joint.point_a.body)
        axis_a = rotation_matrix(pose_a[2]) @ np.asarray(joint.axis_a, dtype=float)
        relative_acceleration = self.point_acceleration(joint.point_b) - self.point_acceleration(joint.point_a)
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
        "_diagnostics",
        "_driver",
        "_joints",
        "_links",
        "_mechanism",
        "_time",
    )

    def __init__(
        self,
        mechanism: Mechanism,
        driver: KinematicDriver,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        *,
        coordinate_velocities: Sequence[Sequence[float]] | np.ndarray | None = None,
        coordinate_accelerations: Sequence[Sequence[float]] | np.ndarray | None = None,
        time: Sequence[float] | np.ndarray | None = None,
        diagnostics: SolveDiagnostics | None = None,
    ) -> None:
        _validate_mechanism(mechanism)
        self._initialize(
            mechanism,
            mechanism.links,
            mechanism.joints,
            driver,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            time=time,
            diagnostics=diagnostics,
        )

    @classmethod
    def _from_snapshot(
        cls,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        joints: tuple[_Joint, ...],
        driver: KinematicDriver,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        *,
        coordinate_velocities: Sequence[Sequence[float]] | np.ndarray | None = None,
        coordinate_accelerations: Sequence[Sequence[float]] | np.ndarray | None = None,
        time: Sequence[float] | np.ndarray | None = None,
        diagnostics: SolveDiagnostics | None = None,
    ) -> KinematicSolution:
        solution = cls.__new__(cls)
        solution._initialize(
            mechanism,
            links,
            joints,
            driver,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            coordinate_accelerations=coordinate_accelerations,
            time=time,
            diagnostics=diagnostics,
        )
        return solution

    def _initialize(
        self,
        mechanism: Mechanism,
        links: tuple[Link, ...],
        joints: tuple[_Joint, ...],
        driver: KinematicDriver,
        coordinates: Sequence[Sequence[float]] | np.ndarray,
        *,
        coordinate_velocities: Sequence[Sequence[float]] | np.ndarray | None,
        coordinate_accelerations: Sequence[Sequence[float]] | np.ndarray | None,
        time: Sequence[float] | np.ndarray | None,
        diagnostics: SolveDiagnostics | None,
    ) -> None:
        _validate_mechanism(mechanism)
        if not isinstance(driver, KinematicDriver):
            raise TypeError("driver must be a KinematicDriver")
        _validate_joint(mechanism, links, driver.joint)
        if coordinate_accelerations is not None and coordinate_velocities is None:
            raise ValueError("coordinate_accelerations requires coordinate_velocities")

        values = driver._position_history()
        coordinate_array = _finite_float_array(coordinates, name="coordinates", ndim=2)
        state_shape = (len(values), 3 * len(links))
        if coordinate_array.shape != state_shape:
            raise ValueError(f"coordinates must have shape {state_shape}")
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
        history_shape = (len(values),)
        time_array = _optional_finite_float_array(
            time,
            name="time",
            shape=history_shape,
        )
        if diagnostics is not None:
            if not isinstance(diagnostics, SolveDiagnostics):
                raise TypeError("diagnostics must be a SolveDiagnostics or None")
            if len(diagnostics) != len(values):
                raise ValueError("diagnostics length must match driver samples")

        self._mechanism = mechanism
        self._links = links
        self._joints = joints
        self._driver = driver
        self._time = time_array
        self._coordinates = coordinate_array
        self._coordinate_velocities = velocity_array
        self._coordinate_accelerations = acceleration_array
        self._diagnostics = diagnostics

    @property
    def mechanism(self) -> Mechanism:
        """Return the mechanism represented by this solution."""
        return self._mechanism

    @property
    def driver(self) -> KinematicDriver:
        """Return the prescribed kinematic driver snapshot."""
        return self._driver

    @property
    def time(self) -> np.ndarray | None:
        """Return physical sample times, if available."""
        if self._time is None:
            return None
        return self._time.copy()

    @property
    def diagnostics(self) -> SolveDiagnostics | None:
        """Return numerical solve diagnostics, if available."""
        return self._diagnostics

    @property
    def coordinates(self) -> np.ndarray:
        """Return a safe copy of the generalized coordinate matrix."""
        return self._coordinates.copy()

    @property
    def coordinate_velocities(self) -> np.ndarray:
        """Return a safe copy of the generalized velocity matrix."""
        if self._coordinate_velocities is None:
            raise ValueError("velocity data is not available in this solution")
        return self._coordinate_velocities.copy()

    @property
    def coordinate_accelerations(self) -> np.ndarray:
        """Return a safe copy of the generalized acceleration matrix."""
        if self._coordinate_accelerations is None:
            raise ValueError("acceleration data is not available in this solution")
        return self._coordinate_accelerations.copy()

    @property
    def has_velocity(self) -> bool:
        """Return whether generalized velocity state is available."""
        return self._coordinate_velocities is not None

    @property
    def has_acceleration(self) -> bool:
        """Return whether generalized acceleration state is available."""
        return self._coordinate_accelerations is not None

    def __len__(self) -> int:
        return len(self._driver)

    def __getitem__(self, index: int | slice) -> Configuration | KinematicSolution:
        """Return one configuration or a sliced kinematic solution."""
        positions = self._driver._position_history()
        velocities = self._driver._velocity_history()
        accelerations = self._driver._acceleration_history()

        if isinstance(index, slice):
            sliced_driver = KinematicDriver._from_history(
                self._driver.joint,
                positions[index],
                None if velocities is None else velocities[index],
                None if accelerations is None else accelerations[index],
            )
            return KinematicSolution._from_snapshot(
                self._mechanism,
                self._links,
                self._joints,
                sliced_driver,
                self._coordinates[index],
                coordinate_velocities=(
                    None if self._coordinate_velocities is None
                    else self._coordinate_velocities[index]
                ),
                coordinate_accelerations=(
                    None if self._coordinate_accelerations is None
                    else self._coordinate_accelerations[index]
                ),
                time=(None if self._time is None else self._time[index]),
                diagnostics=(None if self._diagnostics is None else self._diagnostics._slice(index)),
            )

        item = operator.index(index)
        sample_driver = KinematicDriver(
            self._driver.joint,
            position=float(positions[item]),
            velocity=(None if velocities is None else float(velocities[item])),
            acceleration=(None if accelerations is None else float(accelerations[item])),
        )
        return Configuration._from_snapshot(
            self._mechanism,
            self._links,
            self._coordinates[item],
            coordinate_velocities=(
                None if self._coordinate_velocities is None else self._coordinate_velocities[item]
            ),
            coordinate_accelerations=(
                None if self._coordinate_accelerations is None
                else self._coordinate_accelerations[item]
            ),
            driver=sample_driver,
            time=(None if self._time is None else self._time[item]),
        )

    def __iter__(self) -> Iterator[Configuration]:
        """Iterate over configurations in solution order."""
        for index in range(len(self)):
            yield self[index]

    def point_positions(self, point: Point) -> np.ndarray:
        """Return global point positions for all configurations."""
        _validate_point(self._mechanism, self._links, point)
        values = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].point_position(point)
        return values

    def point_velocities(self, point: Point) -> np.ndarray:
        """Return global point velocities for all configurations."""
        _validate_point(self._mechanism, self._links, point)
        self._require_velocity()
        values = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].point_velocity(point)
        return values

    def point_accelerations(self, point: Point) -> np.ndarray:
        """Return global point accelerations for all configurations."""
        _validate_point(self._mechanism, self._links, point)
        self._require_acceleration()
        values = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].point_acceleration(point)
        return values

    def body_poses(self, body: _Body) -> np.ndarray:
        """Return body pose history with shape ``(N, 3)``."""
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return self._coordinates[:, start : start + 3].copy()

    def body_velocities(self, body: _Body) -> np.ndarray:
        """Return body velocity history with shape ``(N, 3)``."""
        index = _body_index(self._mechanism, self._links, body)
        velocities = self._require_velocity()
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return velocities[:, start : start + 3].copy()

    def body_accelerations(self, body: _Body) -> np.ndarray:
        """Return body acceleration history with shape ``(N, 3)``."""
        index = _body_index(self._mechanism, self._links, body)
        accelerations = self._require_acceleration()
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return accelerations[:, start : start + 3].copy()

    def joint_coordinates(self, joint: _Joint) -> np.ndarray:
        """Return the natural joint coordinate for all configurations."""
        _validate_joint(self._mechanism, self._links, joint)
        values = np.empty(len(self), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].joint_coordinate(joint)
        return values

    def joint_velocities(self, joint: _Joint) -> np.ndarray:
        """Return natural joint velocity for all configurations."""
        _validate_joint(self._mechanism, self._links, joint)
        self._require_velocity()
        values = np.empty(len(self), dtype=float)
        for index in range(len(self)):
            values[index] = self[index].joint_velocity(joint)
        return values

    def joint_accelerations(self, joint: _Joint) -> np.ndarray:
        """Return natural joint acceleration for all configurations."""
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