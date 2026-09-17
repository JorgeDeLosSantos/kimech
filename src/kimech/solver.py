"""High-level kinematic solver for planar mechanisms."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import optimize

from ._constraints import jacobian, residual
from ._differential import solve_acceleration, solve_velocity
from ._scaling import NumericalScaling, build_numerical_scaling
from .errors import InvalidModelError, KinematicSolveError
from .joints import PrismaticJoint, RevoluteJoint
from .model import Link, Mechanism
from .solution import Configuration, KinematicSolution

_Joint = RevoluteJoint | PrismaticJoint
_RESIDUAL_TOL = 1e-9


def solve(
    mechanism: Mechanism,
    *,
    input_joint: RevoluteJoint | PrismaticJoint,
    input_position,
    initial_guess,
    input_velocity=None,
    input_acceleration=None,
) -> KinematicSolution:
    """Solve position and, when requested, differential kinematics."""
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

    if not isinstance(input_joint, (RevoluteJoint, PrismaticJoint)):
        raise TypeError("input_joint must be a RevoluteJoint or PrismaticJoint")
    if not any(input_joint is joint for joint in joints):
        raise InvalidModelError(
            "input_joint does not belong to the mechanism snapshot"
        )

    input_positions, scalar_input = _coerce_input_positions(input_position)
    input_velocities = _coerce_optional_input_history(
        input_velocity,
        name="input_velocity",
        scalar_input=scalar_input,
        count=len(input_positions),
    )
    input_accelerations = _coerce_optional_input_history(
        input_acceleration,
        name="input_acceleration",
        scalar_input=scalar_input,
        count=len(input_positions),
    )
    if input_accelerations is not None and input_velocities is None:
        raise ValueError("input_acceleration requires input_velocity")

    scaling = build_numerical_scaling(
        mechanism,
        links,
        joints,
        input_joint,
        input_positions,
    )
    initial_q = _pack_initial_guess(mechanism, links, initial_guess)

    coordinates = np.empty((len(input_positions), coordinate_count), dtype=float)
    current_guess = initial_q
    for index, input_position_value in enumerate(input_positions):
        accepted = _solve_configuration(
            mechanism,
            links,
            joints,
            input_joint,
            float(input_position_value),
            current_guess,
            scaling,
            input_index=index,
        )
        coordinates[index] = accepted
        current_guess = accepted

    coordinate_velocities = None
    if input_velocities is not None:
        coordinate_velocities = np.empty_like(coordinates)
        for index, (input_position_value, prescribed_velocity) in enumerate(
            zip(input_positions, input_velocities)
        ):
            coordinate_velocities[index] = solve_velocity(
                mechanism,
                links,
                joints,
                input_joint,
                coordinates[index],
                float(input_position_value),
                float(prescribed_velocity),
                scaling,
                input_index=index,
            )

    coordinate_accelerations = None
    if input_accelerations is not None:
        coordinate_accelerations = np.empty_like(coordinates)
        for index, (input_position_value, prescribed_acceleration) in enumerate(
            zip(input_positions, input_accelerations)
        ):
            coordinate_accelerations[index] = solve_acceleration(
                mechanism,
                links,
                joints,
                input_joint,
                coordinates[index],
                coordinate_velocities[index],
                float(input_position_value),
                float(prescribed_acceleration),
                scaling,
                input_index=index,
            )

    return KinematicSolution._from_snapshot(
        mechanism,
        links,
        input_joint,
        input_positions,
        coordinates,
        coordinate_velocities=coordinate_velocities,
        coordinate_accelerations=coordinate_accelerations,
        input_velocities=input_velocities,
        input_accelerations=input_accelerations,
    )


def _coerce_input_positions(input_position: object) -> tuple[np.ndarray, bool]:
    try:
        array = np.asarray(input_position, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("input_position must be numeric") from error

    scalar_input = array.ndim == 0
    if scalar_input:
        array = np.atleast_1d(array)
    elif array.ndim != 1:
        raise ValueError("input_position must be a scalar or a 1-dimensional sequence")
    if array.size == 0:
        raise ValueError("input_position sequence must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("input_position must contain only finite values")
    return array.astype(float, copy=True), scalar_input


def _coerce_optional_input_history(
    value: object,
    *,
    name: str,
    scalar_input: bool,
    count: int,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error

    if array.ndim == 0:
        scalar_value = float(array)
        if not np.isfinite(scalar_value):
            raise ValueError(f"{name} must be finite")
        return np.full(count, scalar_value, dtype=float)

    if scalar_input:
        raise ValueError(f"{name} must be a scalar when input_position is scalar")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1-dimensional sequence")
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
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
    scaling: NumericalScaling,
    *,
    input_index: int | None = None,
) -> np.ndarray:
    def unpack(q_hat: np.ndarray) -> np.ndarray:
        return scaling.unscale_coordinates(q_hat)

    def fun(q_hat: np.ndarray) -> np.ndarray:
        q = unpack(q_hat)
        phi = residual(mechanism, links, joints, input_joint, q, input_value)
        return scaling.scale_residual(phi)

    def jac(q_hat: np.ndarray) -> np.ndarray:
        q = unpack(q_hat)
        matrix = jacobian(mechanism, links, joints, input_joint, q, input_value)
        return scaling.scale_jacobian(matrix)

    initial_q_hat = scaling.scale_coordinates(initial_q)
    result = optimize.root(fun, initial_q_hat, jac=jac, method="hybr")
    expected_shape = (3 * len(links),)
    candidate_hat: np.ndarray | None
    try:
        candidate_hat = np.asarray(result.x, dtype=float)
    except (TypeError, ValueError):
        candidate_hat = None

    residual_norm = float("nan")
    candidate_valid = (
        candidate_hat is not None
        and candidate_hat.shape == expected_shape
        and np.all(np.isfinite(candidate_hat))
    )
    if candidate_valid:
        phi_hat = fun(candidate_hat)
        residual_norm = float(np.linalg.norm(phi_hat, ord=np.inf))

    if (
        candidate_valid
        and np.isfinite(residual_norm)
        and residual_norm <= _RESIDUAL_TOL
    ):
        return unpack(candidate_hat).copy()

    location = (
        f"input index {input_index} (value={input_value:.12g})"
        if input_index is not None
        else f"input value {input_value:.12g}"
    )
    norm_text = f"{residual_norm:.12g}" if np.isfinite(residual_norm) else "unavailable"
    success = bool(getattr(result, "success", False))
    message = getattr(result, "message", "unavailable")
    raise KinematicSolveError(
        f"failed to solve {location}: residual_inf={norm_text}; "
        f"solver success={success}; solver message={message}"
    )
