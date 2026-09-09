"""Private position-constraint equations and analytical Jacobians."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._geometry import perpendicular, rotation_matrix
from .joints import PrismaticJoint, RevoluteJoint
from .model import Ground, Link, Mechanism, Point

_Joint = RevoluteJoint | PrismaticJoint
_Body = Link | Ground


def _relative_axis_angle(axis_a: Sequence[float], axis_b: Sequence[float]) -> float:
    cross = axis_a[0] * axis_b[1] - axis_a[1] * axis_b[0]
    dot = axis_a[0] * axis_b[0] + axis_a[1] * axis_b[1]
    return float(np.arctan2(cross, dot))


def joint_residual(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joint: _Joint,
    q: np.ndarray,
) -> np.ndarray:
    """Evaluate the two geometric equations associated with one joint."""
    coordinates = _validate_context(mechanism, links, joint, q)
    position_a, _, theta_a, _ = _point_kinematics(
        mechanism, links, joint.point_a, coordinates
    )
    position_b, _, theta_b, _ = _point_kinematics(
        mechanism, links, joint.point_b, coordinates
    )

    if isinstance(joint, RevoluteJoint):
        return position_a - position_b

    axis_a = rotation_matrix(theta_a) @ np.asarray(joint.axis_a, dtype=float)
    normal = perpendicular(axis_a)
    displacement = position_b - position_a
    delta_alpha = _relative_axis_angle(joint.axis_a, joint.axis_b)
    return np.array(
        [normal @ displacement, theta_b - theta_a + delta_alpha],
        dtype=float,
    )


def joint_jacobian(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joint: _Joint,
    q: np.ndarray,
) -> np.ndarray:
    """Evaluate the analytical Jacobian of one joint with respect to ``q``."""
    coordinates = _validate_context(mechanism, links, joint, q)
    position_a, index_a, theta_a, angular_a = _point_kinematics(
        mechanism, links, joint.point_a, coordinates
    )
    position_b, index_b, _, angular_b = _point_kinematics(
        mechanism, links, joint.point_b, coordinates
    )
    result = np.zeros((2, 3 * len(links)), dtype=float)

    if isinstance(joint, RevoluteJoint):
        _add_point_block(result, index_a, angular_a, sign=1.0)
        _add_point_block(result, index_b, angular_b, sign=-1.0)
        return result

    axis = rotation_matrix(theta_a) @ np.asarray(joint.axis_a, dtype=float)
    normal = perpendicular(axis)
    displacement = position_b - position_a

    if index_a is not None:
        start = 3 * index_a
        result[0, start : start + 2] = -normal
        result[0, start + 2] = -axis @ displacement - normal @ angular_a
        result[1, start + 2] = -1.0
    if index_b is not None:
        start = 3 * index_b
        result[0, start : start + 2] = normal
        result[0, start + 2] = normal @ angular_b
        result[1, start + 2] = 1.0
    return result


def driver_residual(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joint: _Joint,
    q: np.ndarray,
    value: float,
) -> np.ndarray:
    """Evaluate a joint's natural-coordinate driver equation."""
    coordinates = _validate_context(mechanism, links, joint, q)
    input_value = _finite_scalar(value, name="input_value")
    position_a, _, theta_a, _ = _point_kinematics(
        mechanism, links, joint.point_a, coordinates
    )
    position_b, _, theta_b, _ = _point_kinematics(
        mechanism, links, joint.point_b, coordinates
    )

    if isinstance(joint, RevoluteJoint):
        coordinate = theta_b - theta_a
    else:
        axis = rotation_matrix(theta_a) @ np.asarray(joint.axis_a, dtype=float)
        coordinate = axis @ (position_b - position_a)
    return np.array([coordinate - input_value], dtype=float)


def driver_jacobian(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joint: _Joint,
    q: np.ndarray,
) -> np.ndarray:
    """Evaluate a natural-coordinate driver's analytical Jacobian."""
    coordinates = _validate_context(mechanism, links, joint, q)
    position_a, index_a, theta_a, angular_a = _point_kinematics(
        mechanism, links, joint.point_a, coordinates
    )
    position_b, index_b, _, angular_b = _point_kinematics(
        mechanism, links, joint.point_b, coordinates
    )
    result = np.zeros((1, 3 * len(links)), dtype=float)

    if isinstance(joint, RevoluteJoint):
        if index_a is not None:
            result[0, 3 * index_a + 2] = -1.0
        if index_b is not None:
            result[0, 3 * index_b + 2] = 1.0
        return result

    axis = rotation_matrix(theta_a) @ np.asarray(joint.axis_a, dtype=float)
    normal = perpendicular(axis)
    displacement = position_b - position_a
    if index_a is not None:
        start = 3 * index_a
        result[0, start : start + 2] = -axis
        result[0, start + 2] = normal @ displacement - axis @ angular_a
    if index_b is not None:
        start = 3 * index_b
        result[0, start : start + 2] = axis
        result[0, start + 2] = axis @ angular_b
    return result


def residual(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    q: np.ndarray,
    input_value: float,
) -> np.ndarray:
    """Assemble all joint equations followed by the driver equation."""
    coordinates, value = _validate_system(
        mechanism, links, joints, input_joint, q, input_value
    )
    result = np.empty(2 * len(joints) + 1, dtype=float)
    for index, joint in enumerate(joints):
        result[2 * index : 2 * index + 2] = joint_residual(
            mechanism, links, joint, coordinates
        )
    result[-1:] = driver_residual(mechanism, links, input_joint, coordinates, value)
    return result


def jacobian(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    q: np.ndarray,
    input_value: float,
) -> np.ndarray:
    """Assemble the analytical Jacobian in the same row order as ``residual``."""
    coordinates, _ = _validate_system(
        mechanism, links, joints, input_joint, q, input_value
    )
    result = np.empty((2 * len(joints) + 1, 3 * len(links)), dtype=float)
    for index, joint in enumerate(joints):
        result[2 * index : 2 * index + 2] = joint_jacobian(
            mechanism, links, joint, coordinates
        )
    result[-1:] = driver_jacobian(mechanism, links, input_joint, coordinates)
    return result


def _validate_system(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
    input_joint: _Joint,
    q: object,
    input_value: object,
) -> tuple[np.ndarray, float]:
    _validate_snapshots(mechanism, links, joints)
    if not any(input_joint is joint for joint in joints):
        raise ValueError("input_joint must be included in the joints snapshot")
    coordinates = _finite_coordinates(q, len(links))
    value = _finite_scalar(input_value, name="input_value")
    return coordinates, value


def _validate_context(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joint: object,
    q: object,
) -> np.ndarray:
    _validate_mechanism_and_links(mechanism, links)
    _validate_joint_bodies(mechanism, links, joint)
    return _finite_coordinates(q, len(links))


def _validate_snapshots(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joints: tuple[_Joint, ...],
) -> None:
    _validate_mechanism_and_links(mechanism, links)
    if not isinstance(joints, tuple):
        raise TypeError("joints must be a tuple snapshot")
    for joint in joints:
        _validate_joint_bodies(mechanism, links, joint)
        if not any(joint is owned_joint for owned_joint in mechanism.joints):
            raise ValueError("joint in snapshot does not belong to this mechanism")


def _validate_mechanism_and_links(
    mechanism: object, links: object
) -> None:
    if not isinstance(mechanism, Mechanism):
        raise TypeError("mechanism must be a Mechanism")
    if not isinstance(links, tuple):
        raise TypeError("links must be a tuple snapshot")
    seen: set[int] = set()
    for link in links:
        if not isinstance(link, Link):
            raise TypeError("links must contain only Link objects")
        if link.mechanism is not mechanism:
            raise ValueError("link does not belong to this mechanism")
        if id(link) in seen:
            raise ValueError("links snapshot must not contain duplicates")
        seen.add(id(link))


def _validate_joint_bodies(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    joint: object,
) -> None:
    if not isinstance(joint, (RevoluteJoint, PrismaticJoint)):
        raise TypeError("joint must be a RevoluteJoint or PrismaticJoint")
    _body_index(mechanism, links, joint.point_a.body)
    _body_index(mechanism, links, joint.point_b.body)


def _finite_coordinates(value: object, link_count: int) -> np.ndarray:
    try:
        coordinates = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("q must be numeric") from error
    expected_shape = (3 * link_count,)
    if coordinates.shape != expected_shape:
        raise ValueError(f"q must have shape {expected_shape}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("q must contain only finite values")
    return coordinates


def _finite_scalar(value: object, *, name: str) -> float:
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


def _body_index(
    mechanism: Mechanism, links: tuple[Link, ...], body: _Body
) -> int | None:
    if body is mechanism.ground:
        return None
    for index, link in enumerate(links):
        if body is link:
            return index
    raise ValueError("joint body does not belong to the links snapshot or ground")


def _point_kinematics(
    mechanism: Mechanism,
    links: tuple[Link, ...],
    point: Point,
    q: np.ndarray,
) -> tuple[np.ndarray, int | None, float, np.ndarray]:
    index = _body_index(mechanism, links, point.body)
    if index is None:
        return point.local, None, 0.0, np.zeros(2, dtype=float)
    start = 3 * index
    translation = q[start : start + 2]
    theta = float(q[start + 2])
    rotation = rotation_matrix(theta)
    local = point.local
    return translation + rotation @ local, index, theta, rotation @ perpendicular(local)


def _add_point_block(
    jacobian: np.ndarray,
    index: int | None,
    angular: np.ndarray,
    *,
    sign: float,
) -> None:
    if index is None:
        return
    start = 3 * index
    jacobian[:, start : start + 2] += sign * np.eye(2)
    jacobian[:, start + 2] += sign * angular