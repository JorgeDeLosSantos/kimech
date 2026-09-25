"""Input-coordinate sensitivity analysis for accepted solutions."""

from __future__ import annotations

import numpy as np

from ._differential import solve_input_tangent
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


class InputSensitivity:
    """Sensitivity of a one-input solution with respect to its input coordinate."""

    __slots__ = (
        "_coordinate_derivatives",
        "_coordinates",
        "_input_joint",
        "_input_positions",
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
        self._input_joint = solution.driver.joint
        self._input_positions = solution.driver._position_history().copy()
        self._coordinates = solution.coordinates
        self._coordinate_derivatives = derivatives.copy()

    @property
    def mechanism(self):
        """Return the mechanism associated with this analysis snapshot."""
        return self._mechanism

    @property
    def input_joint(self) -> _Joint:
        """Return the prescribed joint defining the sensitivity coordinate."""
        return self._input_joint

    @property
    def input_positions(self) -> np.ndarray:
        """Return the sampled prescribed coordinate values."""
        return self._input_positions.copy()

    @property
    def coordinate_derivatives(self) -> np.ndarray:
        """Return generalized derivatives dq/du with shape (N, 3*n)."""
        return self._coordinate_derivatives.copy()

    def __len__(self) -> int:
        return len(self._input_positions)

    def body_pose_derivatives(self, body: _Body) -> np.ndarray:
        """Return body-pose derivatives with respect to the input coordinate."""
        index = _body_index(self._mechanism, self._links, body)
        if index is None:
            return np.zeros((len(self), 3), dtype=float)
        start = 3 * index
        return self._coordinate_derivatives[:, start : start + 3].copy()

    def point_position_derivatives(self, point: Point) -> np.ndarray:
        """Return point-position derivatives with respect to the input coordinate."""
        _validate_point(self._mechanism, self._links, point)
        values = np.empty((len(self), 2), dtype=float)
        for index in range(len(self)):
            values[index] = self._configuration(index).point_velocity(point)
        return values

    def joint_coordinate_derivatives(self, joint: _Joint) -> np.ndarray:
        """Return joint-coordinate derivatives with respect to the input coordinate."""
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
                self._input_joint,
                position=float(self._input_positions[index]),
                velocity=1.0,
            ),
        )


def input_sensitivity(solution: KinematicSolution) -> InputSensitivity:
    """Compute local dq/du sensitivity for every accepted solution sample."""
    if not isinstance(solution, KinematicSolution):
        raise TypeError("solution must be a KinematicSolution")

    links = solution._links
    joints = solution._joints
    input_joint = solution.driver.joint
    input_positions = solution.driver._position_history()
    coordinates = solution.coordinates

    driver = KinematicDriver(input_joint, position=input_positions)
    scaling = build_numerical_scaling(
        solution.mechanism,
        links,
        joints,
        driver,
    )

    derivatives = np.empty_like(coordinates)
    for index, (input_value, q) in enumerate(zip(input_positions, coordinates)):
        derivatives[index] = solve_input_tangent(
            solution.mechanism,
            links,
            joints,
            driver,
            q,
            float(input_value),
            scaling,
            input_index=index,
        )

    return InputSensitivity(solution, derivatives)
