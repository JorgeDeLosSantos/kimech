"""High-level kinematic solver for planar mechanisms."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import optimize

from ._constraints import jacobian, residual
from ._differential import solve_acceleration, solve_driver_tangent, solve_velocity
from ._scaling import NumericalScaling, build_numerical_scaling
from .diagnostics import SolveDiagnostics, _jacobian_metrics
from .driver import KinematicDriver
from ._drivers import normalize_drivers
from .errors import InvalidModelError, KinematicSolveError, SolveFailureContext
from .joints import PrismaticJoint, RevoluteJoint
from .model import Link, Mechanism
from .solution import Configuration, KinematicSolution

_Joint = RevoluteJoint | PrismaticJoint
_RESIDUAL_TOL = 1e-9
_MAX_SUBDIVISION_DEPTH = 8


def solve(
    mechanism: Mechanism,
    *,
    driver: KinematicDriver | None = None,
    drivers: KinematicDriver | tuple[KinematicDriver, ...] | list[KinematicDriver] | None = None,
    initial_guess=None,
    time=None,
) -> KinematicSolution:
    """Solve position and, when requested, differential kinematics."""
    if not isinstance(mechanism, Mechanism):
        raise TypeError("mechanism must be a Mechanism")
    if drivers is not None and driver is not None:
        raise TypeError("supply either drivers or legacy driver, not both")
    selected = normalize_drivers(drivers if drivers is not None else driver)
    if len(selected) > 1:
        return _solve_multiple_positions(
            mechanism, selected, initial_guess=initial_guess, time=time
        )
    driver = selected[0]

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
            "solve() with one kinematic driver requires structural mobility 1 "
            f"({coordinate_count} coordinates versus {equation_count} equations)"
        )

    if not any(driver.joint is joint for joint in joints):
        raise InvalidModelError(
            "driver joint does not belong to the mechanism snapshot"
        )

    driver_positions = driver._position_history()
    driver_velocities = driver._velocity_history()
    driver_accelerations = driver._acceleration_history()
    time_values = _coerce_time_history(time, count=len(driver_positions))

    scaling = build_numerical_scaling(
        mechanism,
        links,
        joints,
        driver,
    )
    initial_q = _pack_initial_guess(mechanism, links, initial_guess)

    coordinates = np.empty((len(driver_positions), coordinate_count), dtype=float)
    subdivision_counts = np.zeros(len(driver_positions), dtype=int)
    strategies = np.empty(len(driver_positions), dtype="<U16")
    corrector_attempts = np.zeros(len(driver_positions), dtype=int)
    current_guess = initial_q
    previous_driver_value = None
    previous_accepted = None
    for index, driver_position_value in enumerate(driver_positions):
        driver_value = float(driver_position_value)
        accepted, subdivision_count, strategy, attempts = _solve_requested_configuration(
            mechanism,
            links,
            joints,
            driver,
            driver_value,
            current_guess,
            scaling,
            driver_index=index,
            previous_driver_value=previous_driver_value,
            previous_accepted=previous_accepted,
        )
        coordinates[index] = accepted
        subdivision_counts[index] = subdivision_count
        strategies[index] = strategy
        corrector_attempts[index] = attempts
        previous_driver_value = driver_value
        previous_accepted = accepted

        if index + 1 < len(driver_positions):
            next_driver_value = float(driver_positions[index + 1])
            current_guess = _predict_next_configuration(
                mechanism,
                links,
                joints,
                driver,
                accepted,
                driver_value,
                next_driver_value,
                scaling,
                driver_index=index,
            )

    diagnostics = _build_solve_diagnostics(
        mechanism,
        links,
        joints,
        driver,
        driver_positions,
        coordinates,
        scaling,
        subdivision_counts=subdivision_counts,
        strategies=strategies,
        corrector_attempts=corrector_attempts,
    )

    coordinate_velocities = None
    if driver_velocities is not None:
        coordinate_velocities = np.empty_like(coordinates)
        for index, (driver_position_value, prescribed_velocity) in enumerate(
            zip(driver_positions, driver_velocities)
        ):
            coordinate_velocities[index] = solve_velocity(
                mechanism,
                links,
                joints,
                driver,
                coordinates[index],
                float(driver_position_value),
                float(prescribed_velocity),
                scaling,
                driver_index=index,
            )

    coordinate_accelerations = None
    if driver_accelerations is not None:
        coordinate_accelerations = np.empty_like(coordinates)
        for index, (driver_position_value, prescribed_acceleration) in enumerate(
            zip(driver_positions, driver_accelerations)
        ):
            coordinate_accelerations[index] = solve_acceleration(
                mechanism,
                links,
                joints,
                driver,
                coordinates[index],
                coordinate_velocities[index],
                float(driver_position_value),
                float(prescribed_acceleration),
                scaling,
                driver_index=index,
            )

    return KinematicSolution._from_snapshot(
        mechanism,
        links,
        joints,
        driver,
        coordinates,
        coordinate_velocities=coordinate_velocities,
        coordinate_accelerations=coordinate_accelerations,
        time=time_values,
        diagnostics=diagnostics,
    )


def _solve_requested_configuration(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    driver_value: float,
    preferred_guess: np.ndarray,
    scaling: NumericalScaling,
    *,
    driver_index: int,
    previous_driver_value: float | None,
    previous_accepted: np.ndarray | None,
) -> tuple[np.ndarray, int, str, int]:
    """Solve one requested sample and report its accepted recovery path."""
    attempt_counter = [0]
    first_error: KinematicSolveError | None = None
    preferred_strategy = (
        "initial_guess"
        if previous_accepted is None
        else (
            "warm_start"
            if np.array_equal(preferred_guess, previous_accepted)
            else "predictor"
        )
    )

    try:
        accepted = _attempt_configuration(
            attempt_counter,
            mechanism,
            links,
            joints,
            driver,
            driver_value,
            preferred_guess,
            scaling,
            driver_index=driver_index,
        )
        return accepted, 0, preferred_strategy, attempt_counter[0]
    except KinematicSolveError as error:
        first_error = error

    if previous_accepted is None:
        raise _with_recovery_context(
            first_error,
            attempted_strategies=(preferred_strategy,),
            corrector_attempts=attempt_counter[0],
        )

    if not np.array_equal(preferred_guess, previous_accepted):
        try:
            accepted = _attempt_configuration(
                attempt_counter,
                mechanism,
                links,
                joints,
                driver,
                driver_value,
                previous_accepted,
                scaling,
                driver_index=driver_index,
            )
            return accepted, 0, "warm_start", attempt_counter[0]
        except KinematicSolveError:
            pass

    if previous_driver_value is None or driver_value == previous_driver_value:
        attempted = (
            (preferred_strategy, "warm_start")
            if preferred_strategy != "warm_start"
            else (preferred_strategy,)
        )
        raise _with_recovery_context(
            first_error,
            attempted_strategies=attempted,
            corrector_attempts=attempt_counter[0],
        )

    try:
        accepted, subdivision_count = _solve_with_subdivision(
            mechanism,
            links,
            joints,
            driver,
            start_driver_value=previous_driver_value,
            start_q=previous_accepted,
            target_driver_value=driver_value,
            scaling=scaling,
            driver_index=driver_index,
            depth=0,
            attempt_counter=attempt_counter,
        )
    except KinematicSolveError:
        attempted = (
            (preferred_strategy, "warm_start", "subdivision")
            if preferred_strategy != "warm_start"
            else (preferred_strategy, "subdivision")
        )
        raise _with_recovery_context(
            first_error,
            attempted_strategies=attempted,
            corrector_attempts=attempt_counter[0],
        )
    return accepted, subdivision_count, "subdivision", attempt_counter[0]

def _attempt_configuration(
    attempt_counter: list[int],
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    driver_value: float,
    initial_q: np.ndarray,
    scaling: NumericalScaling,
    *,
    driver_index: int,
) -> np.ndarray:
    """Call the nonlinear corrector while recording one attempted solve."""
    attempt_counter[0] += 1
    return _solve_configuration(
        mechanism,
        links,
        joints,
        driver,
        driver_value,
        initial_q,
        scaling,
        driver_index=driver_index,
    )


def _solve_step_from_accepted(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    *,
    start_driver_value: float,
    start_q: np.ndarray,
    target_driver_value: float,
    scaling: NumericalScaling,
    driver_index: int,
    attempt_counter: list[int],
) -> np.ndarray:
    """Attempt one continuation step using predictor first, then warm start."""
    predicted = _predict_next_configuration(
        mechanism,
        links,
        joints,
        driver,
        start_q,
        start_driver_value,
        target_driver_value,
        scaling,
        driver_index=driver_index,
    )
    try:
        return _attempt_configuration(
            attempt_counter,
            mechanism,
            links,
            joints,
            driver,
            target_driver_value,
            predicted,
            scaling,
            driver_index=driver_index,
        )
    except KinematicSolveError:
        if np.array_equal(predicted, start_q):
            raise
        return _attempt_configuration(
            attempt_counter,
            mechanism,
            links,
            joints,
            driver,
            target_driver_value,
            start_q,
            scaling,
            driver_index=driver_index,
        )


def _solve_with_subdivision(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    *,
    start_driver_value: float,
    start_q: np.ndarray,
    target_driver_value: float,
    scaling: NumericalScaling,
    driver_index: int,
    depth: int,
    attempt_counter: list[int],
) -> tuple[np.ndarray, int]:
    """Recover a failed requested step by recursively bisecting its input interval."""
    if depth >= _MAX_SUBDIVISION_DEPTH:
        raise KinematicSolveError(
            f"adaptive subdivision exhausted at driver index {driver_index} "
            f"(target={target_driver_value:.12g}, depth={depth})",
            context=SolveFailureContext(
                stage="position",
                driver_index=driver_index,
                driver_position=target_driver_value,
            ),
        )

    midpoint = 0.5 * (start_driver_value + target_driver_value)
    if midpoint == start_driver_value or midpoint == target_driver_value:
        raise KinematicSolveError(
            f"adaptive subdivision reached floating-point step limit at driver index "
            f"{driver_index} (target={target_driver_value:.12g})",
            context=SolveFailureContext(
                stage="position",
                driver_index=driver_index,
                driver_position=target_driver_value,
            ),
        )

    try:
        midpoint_q = _solve_step_from_accepted(
            mechanism,
            links,
            joints,
            driver,
            start_driver_value=start_driver_value,
            start_q=start_q,
            target_driver_value=midpoint,
            scaling=scaling,
            driver_index=driver_index,
            attempt_counter=attempt_counter,
        )
    except KinematicSolveError:
        midpoint_q, left_count = _solve_with_subdivision(
            mechanism,
            links,
            joints,
            driver,
            start_driver_value=start_driver_value,
            start_q=start_q,
            target_driver_value=midpoint,
            scaling=scaling,
            driver_index=driver_index,
            depth=depth + 1,
            attempt_counter=attempt_counter,
        )
    else:
        left_count = 0

    try:
        target_q = _solve_step_from_accepted(
            mechanism,
            links,
            joints,
            driver,
            start_driver_value=midpoint,
            start_q=midpoint_q,
            target_driver_value=target_driver_value,
            scaling=scaling,
            driver_index=driver_index,
            attempt_counter=attempt_counter,
        )
        return target_q, left_count + 1
    except KinematicSolveError:
        target_q, right_count = _solve_with_subdivision(
            mechanism,
            links,
            joints,
            driver,
            start_driver_value=midpoint,
            start_q=midpoint_q,
            target_driver_value=target_driver_value,
            scaling=scaling,
            driver_index=driver_index,
            depth=depth + 1,
            attempt_counter=attempt_counter,
        )
        return target_q, left_count + 1 + right_count


def _predict_next_configuration(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    q: np.ndarray,
    driver_value: float,
    next_driver_value: float,
    scaling: NumericalScaling,
    *,
    driver_index: int | None = None,
) -> np.ndarray:
    """Return a first-order continuation predictor, falling back to warm start."""
    delta_driver = next_driver_value - driver_value
    if delta_driver == 0.0:
        return q.copy()

    try:
        tangent = solve_driver_tangent(
            mechanism,
            links,
            joints,
            driver,
            q,
            driver_value,
            scaling,
            driver_index=driver_index,
        )
    except KinematicSolveError:
        return q.copy()

    predicted = q + delta_driver * tangent
    if predicted.shape != q.shape or not np.all(np.isfinite(predicted)):
        return q.copy()
    return predicted


def _build_solve_diagnostics(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    driver_positions: np.ndarray,
    coordinates: np.ndarray,
    scaling: NumericalScaling,
    *,
    subdivision_counts: np.ndarray | None = None,
    strategies: np.ndarray | None = None,
    corrector_attempts: np.ndarray | None = None,
) -> SolveDiagnostics:
    count = len(driver_positions)
    condition_numbers = np.empty(count, dtype=float)
    min_singular_values = np.empty(count, dtype=float)
    ranks = np.empty(count, dtype=int)
    residual_norms = np.empty(count, dtype=float)

    for index, (driver_value, q) in enumerate(zip(driver_positions, coordinates)):
        matrix = jacobian(
            mechanism,
            links,
            joints,
            driver,
            q,
            float(driver_value),
        )
        phi = residual(
            mechanism,
            links,
            joints,
            driver,
            q,
            float(driver_value),
        )
        residual_norms[index] = float(
            np.linalg.norm(scaling.scale_residual(phi), ord=np.inf)
        )
        matrix_hat = scaling.scale_jacobian(matrix)
        condition, minimum, rank = _jacobian_metrics(matrix_hat)
        condition_numbers[index] = condition
        min_singular_values[index] = minimum
        ranks[index] = rank

    return SolveDiagnostics(
        condition_numbers,
        min_singular_values,
        ranks,
        subdivision_counts=subdivision_counts,
        strategies=strategies,
        corrector_attempts=corrector_attempts,
        residual_norms=residual_norms,
    )



def _coerce_time_history(time: object, *, count: int) -> np.ndarray | None:
    """Validate optional physical sample times for one solve request."""
    if time is None:
        return None
    try:
        array = np.asarray(time, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("time must be numeric") from error

    if array.ndim == 0:
        if count != 1:
            raise ValueError(f"time must have shape ({count},)")
        array = np.atleast_1d(array)
    elif array.ndim != 1:
        raise ValueError("time must be a scalar or a 1-dimensional sequence")

    if array.shape != (count,):
        raise ValueError(f"time must have shape ({count},)")
    if not np.all(np.isfinite(array)):
        raise ValueError("time must contain only finite values")
    if count > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError("time must be strictly increasing")
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


def _with_recovery_context(
    error: KinematicSolveError,
    *,
    attempted_strategies: tuple[str, ...],
    corrector_attempts: int,
) -> KinematicSolveError:
    """Return an equivalent failure enriched with requested-sample recovery data."""
    context = error.context
    if context is None:
        return KinematicSolveError(str(error))
    return KinematicSolveError(
        str(error),
        context=SolveFailureContext(
            stage=context.stage,
            driver_index=context.driver_index,
            driver_position=context.driver_position,
            residual_norm=context.residual_norm,
            condition_number=context.condition_number,
            min_singular_value=context.min_singular_value,
            rank=context.rank,
            attempted_strategies=attempted_strategies,
            corrector_attempts=corrector_attempts,
        ),
    )


def _solve_configuration(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    driver: KinematicDriver,
    driver_value: float,
    initial_q: np.ndarray,
    scaling: NumericalScaling,
    *,
    driver_index: int | None = None,
) -> np.ndarray:
    def unpack(q_hat: np.ndarray) -> np.ndarray:
        return scaling.unscale_coordinates(q_hat)

    def fun(q_hat: np.ndarray) -> np.ndarray:
        q = unpack(q_hat)
        phi = residual(mechanism, links, joints, driver, q, driver_value)
        return scaling.scale_residual(phi)

    def jac(q_hat: np.ndarray) -> np.ndarray:
        q = unpack(q_hat)
        matrix = jacobian(mechanism, links, joints, driver, q, driver_value)
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
        f"driver index {driver_index} (value={driver_value:.12g})"
        if driver_index is not None
        else f"driver value {driver_value:.12g}"
    )
    norm_text = f"{residual_norm:.12g}" if np.isfinite(residual_norm) else "unavailable"
    success = bool(getattr(result, "success", False))
    message = getattr(result, "message", "unavailable")
    condition_number = None
    min_singular_value = None
    rank = None
    if candidate_valid:
        try:
            matrix_hat = jac(candidate_hat)
            condition_number, min_singular_value, rank = _jacobian_metrics(matrix_hat)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass

    raise KinematicSolveError(
        f"failed to solve {location}: residual_inf={norm_text}; "
        f"solver success={success}; solver message={message}",
        context=SolveFailureContext(
            stage="position",
            driver_index=driver_index,
            driver_position=driver_value,
            residual_norm=(
                residual_norm if np.isfinite(residual_norm) else None
            ),
            condition_number=condition_number,
            min_singular_value=min_singular_value,
            rank=rank,
        ),
    )


def _solve_multiple_positions(
    mechanism: Mechanism,
    drivers: tuple[KinematicDriver, ...],
    *,
    initial_guess,
    time,
) -> KinematicSolution:
    """Solve fully prescribed multi-driver kinematics along an input-space path.

    Positions are determined first; differential states use the same scaled
    analytical Jacobian at each accepted configuration.
    """
    links = mechanism.links
    joints = mechanism.joints
    report = mechanism.validate()
    if not report.is_valid:
        raise InvalidModelError("invalid mechanism: " + "; ".join(report.errors))
    n = 3 * len(links)
    equations = 2 * len(joints) + len(drivers)
    if equations != n:
        raise InvalidModelError(
            f"solve() requires {n - 2 * len(joints)} independent drivers "
            f"({n} coordinates versus {equations} equations)"
        )
    for selected in drivers:
        if not any(selected.joint is joint for joint in joints):
            raise InvalidModelError("driver joint does not belong to the mechanism snapshot")

    velocity_histories = tuple(item._velocity_history() for item in drivers)
    acceleration_histories = tuple(item._acceleration_history() for item in drivers)
    has_any_velocity = any(history is not None for history in velocity_histories)
    has_all_velocity = all(history is not None for history in velocity_histories)
    has_any_acceleration = any(history is not None for history in acceleration_histories)
    has_all_acceleration = all(history is not None for history in acceleration_histories)
    if has_any_velocity and not has_all_velocity:
        raise ValueError("all drivers must prescribe velocity to solve Multi-DOF velocities")
    if has_any_acceleration and not has_all_acceleration:
        raise ValueError("all drivers must prescribe acceleration to solve Multi-DOF accelerations")

    inputs = np.column_stack([item._position_history() for item in drivers])
    input_velocities = (
        np.column_stack(velocity_histories) if has_all_velocity else None
    )
    input_accelerations = (
        np.column_stack(acceleration_histories) if has_all_acceleration else None
    )
    time_values = _coerce_time_history(time, count=len(inputs))
    scaling = build_numerical_scaling(mechanism, links, joints, drivers)
    q0 = _pack_initial_guess(mechanism, links, initial_guess)
    states = np.empty((len(inputs), n), dtype=float)
    subdivisions = np.zeros(len(inputs), dtype=int)
    strategies = np.empty(len(inputs), dtype="<U16")
    attempts = np.zeros(len(inputs), dtype=int)

    def correct(initial_q: np.ndarray, values: np.ndarray) -> np.ndarray:
        attempts_counter[0] += 1
        def fun(q_hat: np.ndarray) -> np.ndarray:
            return scaling.scale_residual(
                residual(mechanism, links, joints, drivers,
                         scaling.unscale_coordinates(q_hat), values)
            )
        def jac(q_hat: np.ndarray) -> np.ndarray:
            return scaling.scale_jacobian(
                jacobian(mechanism, links, joints, drivers,
                         scaling.unscale_coordinates(q_hat), values)
            )
        result = optimize.root(fun, scaling.scale_coordinates(initial_q),
                               jac=jac, method="hybr")
        candidate = np.asarray(result.x, dtype=float)
        if candidate.shape == (n,) and np.all(np.isfinite(candidate)):
            if np.linalg.norm(fun(candidate), ord=np.inf) <= _RESIDUAL_TOL:
                return scaling.unscale_coordinates(candidate).copy()
        raise KinematicSolveError(
            "failed to solve multi-driver position configuration: "
            + str(getattr(result, "message", "non-convergent residual"))
        )

    def predict(q: np.ndarray, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        if np.array_equal(start, target):
            return q.copy()
        matrix = jacobian(mechanism, links, joints, drivers, q, start)
        rhs = np.concatenate((np.zeros(2 * len(joints)), target - start))
        try:
            delta_hat = np.linalg.solve(
                scaling.scale_jacobian(matrix), scaling.scale_rhs(rhs)
            )
            predicted = q + scaling.unscale_state(delta_hat)
            if np.all(np.isfinite(predicted)):
                return predicted
        except np.linalg.LinAlgError:
            pass
        return q.copy()

    def step(q: np.ndarray, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        guess = predict(q, start, target)
        try:
            return correct(guess, target)
        except KinematicSolveError:
            if np.array_equal(guess, q):
                raise
            return correct(q, target)

    def subdivide(q: np.ndarray, start: np.ndarray,
                  target: np.ndarray, depth: int) -> tuple[np.ndarray, int]:
        if depth >= _MAX_SUBDIVISION_DEPTH:
            raise KinematicSolveError(
                f"multi-driver adaptive subdivision exhausted (depth={depth})"
            )
        midpoint = 0.5 * start + 0.5 * target
        if np.array_equal(midpoint, start) or np.array_equal(midpoint, target):
            raise KinematicSolveError("multi-driver subdivision reached floating-point limit")
        try:
            qmid = step(q, start, midpoint)
            left_count = 0
        except KinematicSolveError:
            qmid, left_count = subdivide(q, start, midpoint, depth + 1)
        try:
            target_q = step(qmid, midpoint, target)
            return target_q, left_count + 1
        except KinematicSolveError:
            target_q, right_count = subdivide(qmid, midpoint, target, depth + 1)
            return target_q, left_count + right_count + 1

    previous_q = None
    previous_u = None
    for index, target in enumerate(inputs):
        attempts_counter = [0]
        if previous_q is None:
            try:
                accepted = correct(q0, target)
            except KinematicSolveError as error:
                raise KinematicSolveError(
                    f"multi-driver position solve failed at sample {index}: {error}"
                ) from error
            strategy = "initial_guess"
            count = 0
        else:
            guess = predict(previous_q, previous_u, target)
            try:
                accepted = correct(guess, target)
                strategy = "warm_start" if np.array_equal(guess, previous_q) else "predictor"
                count = 0
            except KinematicSolveError:
                try:
                    accepted = correct(previous_q, target)
                    strategy = "warm_start"
                    count = 0
                except KinematicSolveError:
                    try:
                        accepted, count = subdivide(previous_q, previous_u, target, 0)
                        strategy = "subdivision"
                    except KinematicSolveError as error:
                        raise KinematicSolveError(
                            f"multi-driver position solve failed at sample {index}: {error}"
                        ) from error
        states[index] = accepted
        previous_q, previous_u = accepted, target.copy()
        subdivisions[index] = count
        strategies[index] = strategy
        attempts[index] = attempts_counter[0]

    conditions = np.empty(len(inputs))
    minima = np.empty(len(inputs))
    ranks = np.empty(len(inputs), dtype=int)
    norms = np.empty(len(inputs))
    for index, (q, values) in enumerate(zip(states, inputs)):
        matrix = scaling.scale_jacobian(jacobian(
            mechanism, links, joints, drivers, q, values
        ))
        conditions[index], minima[index], ranks[index] = _jacobian_metrics(matrix)
        norms[index] = np.linalg.norm(scaling.scale_residual(
            residual(mechanism, links, joints, drivers, q, values)
        ), ord=np.inf)

    diagnostics = SolveDiagnostics(
        conditions, minima, ranks, subdivision_counts=subdivisions,
        strategies=strategies, corrector_attempts=attempts, residual_norms=norms,
    )
    coordinate_velocities = None
    if input_velocities is not None:
        coordinate_velocities = np.empty_like(states)
        for index, (q, prescribed_positions, prescribed_rates) in enumerate(
            zip(states, inputs, input_velocities)
        ):
            coordinate_velocities[index] = solve_velocity(
                mechanism, links, joints, drivers, q, prescribed_positions,
                prescribed_rates, scaling, driver_index=index,
            )

    coordinate_accelerations = None
    if input_accelerations is not None:
        coordinate_accelerations = np.empty_like(states)
        for index, (q, q_dot, prescribed_positions, prescribed_accelerations) in enumerate(
            zip(states, coordinate_velocities, inputs, input_accelerations)
        ):
            coordinate_accelerations[index] = solve_acceleration(
                mechanism, links, joints, drivers, q, q_dot, prescribed_positions,
                prescribed_accelerations, scaling, driver_index=index,
            )

    return KinematicSolution._from_snapshot(
        mechanism, links, joints, drivers, states,
        coordinate_velocities=coordinate_velocities,
        coordinate_accelerations=coordinate_accelerations,
        time=time_values, diagnostics=diagnostics,
    )
