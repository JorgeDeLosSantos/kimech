"""High-level kinematic solver for planar mechanisms."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import optimize

from ._constraints import jacobian, residual
from ._differential import solve_velocity
from .errors import InvalidModelError, KinematicSolveError
from .joints import PrismaticJoint, RevoluteJoint
from .model import Link, Mechanism
from .solution import Configuration, KinematicSolution

_Joint = RevoluteJoint | PrismaticJoint
_RESIDUAL_TOL = 1e-9


def solve(
    mechanism: Mechanism,
    *,
    input: RevoluteJoint | PrismaticJoint,
    values,
    initial_guess,
    input_velocity=None,
) -> Configuration | KinematicSolution:
    """Solve position and, when requested, velocity kinematics."""
    if not isinstance(mechanism, Mechanism):
        raise TypeError("mechanism must be a Mechanism")

    links = mechanism.links
    joints = mechanism.joints

    report = mechanism.validate()
    if not report.is_valid:
        details = "; ".join(report.errors)
        raise InvalidModelError(f"invalid mechanism: {details}")

    coordinate_count = 3 * len(links)
    equation_count = 2 * len(joints) + 1
    if coordinate_count != equation_count:
        raise InvalidModelError(
            "solve() with one prescribed input requires structural mobility 1 "
            f"({coordinate_count} coordinates versus {equation_count} equations)"
        )

    if not isinstance(input, (RevoluteJoint, PrismaticJoint)):
        raise TypeError("input must be a RevoluteJoint or PrismaticJoint")
    if not any(input is joint for joint in joints):
        raise InvalidModelError("input joint does not belong to the mechanism snapshot")

    input_values, scalar = _coerce_input_values(values)
    input_velocities = _coerce_input_velocity(
        input_velocity,
        scalar=scalar,
        count=len(input_values),
    )
    initial_q = _pack_initial_guess(mechanism, links, initial_guess)

    if scalar:
        input_value = float(input_values[0])
        coordinates = _solve_configuration(
            mechanism,
            links,
            joints,
            input,
            input_value,
            initial_q,
        )
        coordinate_velocities = None
        prescribed_velocity = None
        if input_velocities is not None:
            prescribed_velocity = float(input_velocities[0])
            coordinate_velocities = solve_velocity(
                mechanism,
                links,
                joints,
                input,
                coordinates,
                input_value,
                prescribed_velocity,
            )
        return Configuration._from_snapshot(
            mechanism,
            links,
            coordinates,
            coordinate_velocities=coordinate_velocities,
            input_joint=input,
            input_value=input_value,
            input_velocity=prescribed_velocity,
        )

    coordinates = np.empty((len(input_values), coordinate_count), dtype=float)
    current_guess = initial_q
    for index, input_value in enumerate(input_values):
        accepted = _solve_configuration(
            mechanism,
            links,
            joints,
            input,
            float(input_value),
            current_guess,
            input_index=index,
        )
        coordinates[index] = accepted
        current_guess = accepted

    coordinate_velocities = None
    if input_velocities is not None:
        coordinate_velocities = np.empty_like(coordinates)
        for index, (input_value, prescribed_velocity) in enumerate(
            zip(input_values, input_velocities)
        ):
            coordinate_velocities[index] = solve_velocity(
                mechanism,
                links,
                joints,
                input,
                coordinates[index],
                float(input_value),
                float(prescribed_velocity),
                input_index=index,
            )

    return KinematicSolution._from_snapshot(
        mechanism,
        links,
        input,
        input_values,
        coordinates,
        coordinate_velocities=coordinate_velocities,
        input_velocities=input_velocities,
    )


def _coerce_input_values(values: object) -> tuple[np.ndarray, bool]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("values must be numeric") from error

    scalar = array.ndim == 0
    if not scalar and array.ndim != 1:
        raise ValueError("values must be a scalar or a 1-dimensional sequence")
    if not scalar and array.size == 0:
        raise ValueError("values sequence must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite values")

    return np.atleast_1d(array).astype(float, copy=True), scalar


def _coerce_input_velocity(
    value: object,
    *,
    scalar: bool,
    count: int,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("input_velocity must be numeric") from error

    if scalar:
        if array.ndim != 0:
            raise ValueError("input_velocity must be a scalar when values is scalar")
        if not np.isfinite(float(array)):
            raise ValueError("input_velocity must be finite")
        return np.array([float(array)], dtype=float)

    if array.ndim == 0:
        scalar_value = float(array)
        if not np.isfinite(scalar_value):
            raise ValueError("input_velocity must be finite")
        return np.full(count, scalar_value, dtype=float)

    if array.ndim != 1:
        raise ValueError("input_velocity must be a scalar or a 1-dimensional sequence")
    if array.shape != (count,):
        raise ValueError(f"input_velocity must have shape ({count},)")
    if not np.all(np.isfinite(array)):
        raise ValueError("input_velocity must contain only finite values")
    return array.astype(float, copy=True)


def _pack_initial_guess(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    initial_guess: object,
) -> np.ndarray:
    if isinstance(initial_guess, Configuration):
        if initial_guess.mechanism is not mechanism:
            raise ValueError("initial_guess Configuration belongs to another mechanism")
        poses: list[object] = []
        for link in links:
            try:
                poses.append(initial_guess.body_pose(link))
            except ValueError as error:
                raise ValueError(
                    "initial_guess Configuration is incompatible with the links snapshot"
                ) from error
        return _pack_poses(poses)

    if not isinstance(initial_guess, Mapping):
        raise TypeError("initial_guess must be a Mapping or Configuration")

    keys = list(initial_guess.keys())
    has_every_link = all(any(key is link for key in keys) for link in links)
    has_only_links = all(any(key is link for link in links) for key in keys)
    if len(keys) != len(links) or not has_every_link or not has_only_links:
        raise ValueError("initial_guess mapping must contain exactly all links in the snapshot")

    return _pack_poses([initial_guess[link] for link in links])


def _pack_poses(poses: list[object]) -> np.ndarray:
    packed = np.empty(3 * len(poses), dtype=float)
    for index, pose in enumerate(poses):
        try:
            array = np.asarray(pose, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("initial_guess poses must be numeric") from error
        if array.shape != (3,):
            raise ValueError("each initial_guess pose must have shape (3,)")
        if not np.all(np.isfinite(array)):
            raise ValueError("initial_guess poses must contain only finite values")
        packed[3 * index : 3 * index + 3] = array
    return packed


def _solve_configuration(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    input_value: float,
    initial_q: np.ndarray,
    *,
    input_index: int | None = None,
) -> np.ndarray:
    def fun(q: np.ndarray) -> np.ndarray:
        return residual(mechanism, links, joints, input_joint, q, input_value)

    def jac(q: np.ndarray) -> np.ndarray:
        return jacobian(mechanism, links, joints, input_joint, q, input_value)

    result = optimize.root(fun, initial_q, jac=jac, method="hybr")
    expected_shape = (3 * len(links),)
    candidate: np.ndarray | None
    try:
        candidate = np.asarray(result.x, dtype=float)
    except (TypeError, ValueError):
        candidate = None

    residual_norm = float("nan")
    candidate_valid = (
        candidate is not None
        and candidate.shape == expected_shape
        and np.all(np.isfinite(candidate))
    )
    if candidate_valid:
        phi = residual(
            mechanism,
            links,
            joints,
            input_joint,
            candidate,
            input_value,
        )
        residual_norm = float(np.linalg.norm(phi, ord=np.inf))

    if (
        result.success is True
        and candidate_valid
        and np.isfinite(residual_norm)
        and residual_norm <= _RESIDUAL_TOL
    ):
        return candidate.copy()

    location = (
        f"input index {input_index} (value={input_value:.12g})"
        if input_index is not None
        else f"input value {input_value:.12g}"
    )
    norm_text = f"{residual_norm:.12g}" if np.isfinite(residual_norm) else "unavailable"
    raise KinematicSolveError(
        f"failed to solve {location}: residual_inf={norm_text}; "
        f"solver message={result.message}"
    )
