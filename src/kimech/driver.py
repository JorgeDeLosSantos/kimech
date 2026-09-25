"""Prescribed kinematic drivers."""

from __future__ import annotations

import numpy as np

from .joints import PrismaticJoint, RevoluteJoint

_Joint = RevoluteJoint | PrismaticJoint


class KinematicDriver:
    """Prescribe one joint natural coordinate and optional time derivatives.

    Kimech 0.6.0 initially supports the natural coordinate of one revolute or
    prismatic joint.  The driver stores kinematic data only; physical sample
    times, numerical continuation, and solver configuration belong elsewhere.
    """

    __slots__ = (
        "_acceleration_is_scalar",
        "_accelerations",
        "_joint",
        "_position_is_scalar",
        "_positions",
        "_velocity_is_scalar",
        "_velocities",
    )

    def __init__(
        self,
        joint: _Joint,
        *,
        position,
        velocity=None,
        acceleration=None,
    ) -> None:
        if not isinstance(joint, (RevoluteJoint, PrismaticJoint)):
            raise TypeError("joint must be a RevoluteJoint or PrismaticJoint")

        positions, position_is_scalar = _coerce_positions(position)
        velocities, velocity_is_scalar = _coerce_optional_history(
            velocity,
            name="velocity",
            scalar_position=position_is_scalar,
            count=len(positions),
        )
        accelerations, acceleration_is_scalar = _coerce_optional_history(
            acceleration,
            name="acceleration",
            scalar_position=position_is_scalar,
            count=len(positions),
        )
        if accelerations is not None and velocities is None:
            raise ValueError("acceleration requires velocity")

        object.__setattr__(self, "_joint", joint)
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_position_is_scalar", position_is_scalar)
        object.__setattr__(self, "_velocities", velocities)
        object.__setattr__(self, "_velocity_is_scalar", velocity_is_scalar)
        object.__setattr__(self, "_accelerations", accelerations)
        object.__setattr__(self, "_acceleration_is_scalar", acceleration_is_scalar)

    def __setattr__(self, name, value) -> None:
        raise AttributeError("KinematicDriver is immutable")

    @property
    def joint(self) -> _Joint:
        """Return the joint whose natural coordinate is prescribed."""
        return self._joint

    @property
    def position(self) -> float | np.ndarray:
        """Return the prescribed position scalar or a safe history copy."""
        if self._position_is_scalar:
            return float(self._positions[0])
        return self._positions.copy()

    @property
    def velocity(self) -> float | np.ndarray | None:
        """Return prescribed velocity data, if available."""
        if self._velocities is None:
            return None
        if self._velocity_is_scalar:
            return float(self._velocities[0])
        return self._velocities.copy()

    @property
    def acceleration(self) -> float | np.ndarray | None:
        """Return prescribed acceleration data, if available."""
        if self._accelerations is None:
            return None
        if self._acceleration_is_scalar:
            return float(self._accelerations[0])
        return self._accelerations.copy()

    @property
    def sample_count(self) -> int:
        """Return the number of requested position samples."""
        return len(self._positions)

    @property
    def is_scalar(self) -> bool:
        """Return whether the prescribed position was supplied as a scalar."""
        return self._position_is_scalar

    def __len__(self) -> int:
        return len(self._positions)

    def _position_history(self) -> np.ndarray:
        return self._positions

    def _velocity_history(self) -> np.ndarray | None:
        return self._velocities

    def _acceleration_history(self) -> np.ndarray | None:
        return self._accelerations


def _coerce_positions(value: object) -> tuple[np.ndarray, bool]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("position must be numeric") from error

    is_scalar = array.ndim == 0
    if is_scalar:
        array = np.atleast_1d(array)
    elif array.ndim != 1:
        raise ValueError("position must be a scalar or a 1-dimensional sequence")
    if array.size == 0:
        raise ValueError("position sequence must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("position must contain only finite values")
    return array.astype(float, copy=True), is_scalar


def _coerce_optional_history(
    value: object,
    *,
    name: str,
    scalar_position: bool,
    count: int,
) -> tuple[np.ndarray | None, bool]:
    if value is None:
        return None, False

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error

    if array.ndim == 0:
        scalar = float(array)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return np.full(count, scalar, dtype=float), True

    if scalar_position:
        raise ValueError(f"{name} must be a scalar when position is scalar")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1-dimensional sequence")
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(float, copy=True), False
