"""Driver-coordinate sensitivity analysis for accepted solutions."""

from __future__ import annotations

import numpy as np

from ._differential import solve_driver_tangent
from ._scaling import build_numerical_scaling
from .driver import KinematicDriver
from .joints import PrismaticJoint, RevoluteJoint
from .model import Ground, Link, Point
from .solution import (
    Configuration,
    KinematicSolution,
    _body_index,
    _validate_joint,
    _validate_point,
)

_Joint = RevoluteJoint | PrismaticJoint
_Body = Link | Ground


class DriverSensitivity:
    """Sensitivity of a driven solution with respect to its driver coordinate."""

    __slots__ = (
        "_coordinate_derivatives",
        "_coordinates",
        "_driver",
        "_driver_positions",
        "_joints",
        "_links",
        "_mechanism",
    )

    def __init__(
        self,
        solution: KinematicSolution,
        coordinate_derivatives: np.ndarray,
    ) -> None:
        if not isinstance(solution, KinematicSolution):
            raise TypeError("solution must be a KinematicSolution")

        derivatives = np.asarray(coordinate_derivatives, dtype=float)
        expected_shape = solution.coordinates.shape
        if derivatives.shape != expected_shape:
            raise ValueError(
                f"coordinate_derivatives must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(derivatives)):
            raise ValueError("coordinate_derivatives must contain only finite values")

        self._mechanism = solution.mechanism
        self._links = solution._links
        self._joints = solution._joints
        self._driver = solution.driver
        self._driver_positions = solution.driver._position_history().copy()
        self._coordinates = solution.coordinates
        self._coordinate_derivatives = derivatives.copy()

    @property
    def mechanism(self):
        """Return the mechanism associated with this analysis snapshot."""
        return self._mechanism

    @property
    def driver(self) -> KinematicDriver:
        """Return the driver defining the sensitivity coordinate."""
        return self._driver

    @property
    def coordinate_derivatives(self) -> np.ndarray:
        """Return generalized derivatives dq/du with shape (N, 3*n)."""
        return self._coordinate_derivatives.copy()

    def __len__(self) -> int:
        return len(self._driver_positions)

    def body_pose_derivatives(self, body: _Body) -> np.ndarray:
        """Return body-pose derivatives with respect to the driver coordinate."""
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return self._coordinate_derivatives[:, start : start + 3].copy()

    def point_position_derivatives(self, point: Point) -> np.ndarray:
        """Return point-position derivatives with respect to the driver coordinate."""
        _validate_point(self._mechanism, self._links, point)
        values = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            values[index] = self._configuration(index).point_velocity(point)
        return values

    def joint_coordinate_derivatives(self, joint: _Joint) -> np.ndarray:
        """Return joint-coordinate derivatives with respect to the driver coordinate."""
        if not isinstance(joint, (RevoluteJoint, PrismaticJoint)):
            raise TypeError("joint must be a RevoluteJoint or PrismaticJoint")
        if not any(joint is item for item in self._joints):
            raise ValueError("joint does not belong to this sensitivity snapshot")
        _validate_joint(self._mechanism, self._links, joint)

        values = np.empty(len(self), dtype=float)
        for index in range(len(self)):
            values[index] = self._configuration(index).joint_velocity(joint)
        return values

    def _configuration(self, index: int) -> Configuration:
        return Configuration._from_snapshot(
            self._mechanism,
            self._links,
            self._coordinates[index],
            coordinate_velocities=self._coordinate_derivatives[index],
            driver=KinematicDriver(
                self._driver.joint,
                position=float(self._driver_positions[index]),
                velocity=1.0,
            ),
        )


def driver_sensitivity(solution: KinematicSolution) -> DriverSensitivity:
    """Compute local dq/du sensitivity for every accepted driver sample."""
    if not isinstance(solution, KinematicSolution):
        raise TypeError("solution must be a KinematicSolution")

    links = solution._links
    joints = solution._joints
    driver = solution.driver
    driver_positions = driver._position_history()
    coordinates = solution.coordinates

    scaling = build_numerical_scaling(
        solution.mechanism,
        links,
        joints,
        driver,
    )

    derivatives = np.empty_like(coordinates)
    for index, (driver_value, q) in enumerate(zip(driver_positions, coordinates)):
        derivatives[index] = solve_driver_tangent(
            solution.mechanism,
            links,
            joints,
            driver,
            q,
            float(driver_value),
            scaling,
            driver_index=index,
        )

    return DriverSensitivity(solution, derivatives)
