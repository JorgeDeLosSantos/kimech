import numpy as np
import pytest

from kimech import Mechanism
from kimech._constraints import (
    driver_acceleration_bias,
    driver_jacobian,
    joint_acceleration_bias,
    joint_jacobian,
)


def _directional_jacobian_rate(jacobian_func, q, q_dot, eps=1e-7):
    q = np.asarray(q, dtype=float)
    q_dot = np.asarray(q_dot, dtype=float)
    scale = max(1.0, np.linalg.norm(q_dot, ord=np.inf))
    step = eps / scale
    return (jacobian_func(q + step * q_dot) - jacobian_func(q - step * q_dot)) / (2.0 * step)


def _revolute_case(kind):
    mechanism = Mechanism()
    body_a = mechanism.add_link("a")
    point_a = body_a.add_point("A", (0.6, -0.35))
    body_b = mechanism.add_link("b")
    point_b = body_b.add_point("B", (-0.45, 0.7))
    ground_a = mechanism.ground.add_point("GA", (0.2, -0.8))
    ground_b = mechanism.ground.add_point("GB", (-0.3, 1.1))
    if kind == "ground-mobile":
        joint = mechanism.revolute(ground_a, point_b)
    elif kind == "mobile-ground":
        joint = mechanism.revolute(point_a, ground_b)
    else:
        joint = mechanism.revolute(point_a, point_b)
    q = np.array([0.3, -0.9, 0.43, 1.7, 0.6, -0.52])
    q_dot = np.array([0.8, -0.4, 1.3, -0.2, 0.7, -0.9])
    return mechanism, joint, q, q_dot


def _prismatic_case(kind):
    mechanism = Mechanism()
    body_a = mechanism.add_link("a")
    point_a = body_a.add_point("A", (0.6, -0.35))
    body_b = mechanism.add_link("b")
    point_b = body_b.add_point("B", (-0.45, 0.7))
    ground = mechanism.ground.add_point("G", (0.2, -0.8))
    if kind == "ground-mobile":
        joint = mechanism.prismatic(
            ground,
            point_b,
            axis_a=(0.6, 0.8),
            axis_b=(-0.2, 0.98),
        )
    elif kind == "mobile-ground":
        joint = mechanism.prismatic(
            point_a,
            ground,
            axis_a=(0.6, 0.8),
            axis_b=(-0.2, 0.98),
        )
    else:
        joint = mechanism.prismatic(
            point_a,
            point_b,
            axis_a=(0.6, 0.8),
            axis_b=(0.9, -0.3),
        )
    q = np.array([0.3, -0.9, 0.43, 1.7, 0.6, -0.52])
    q_dot = np.array([0.8, -0.4, 1.3, -0.2, 0.7, -0.9])
    return mechanism, joint, q, q_dot


@pytest.mark.parametrize("kind", ["ground-mobile", "mobile-ground", "mobile-mobile"])
def test_revolute_joint_acceleration_bias_matches_directional_jacobian_rate(kind):
    mechanism, joint, q, q_dot = _revolute_case(kind)
    links = mechanism.links
    numerical = _directional_jacobian_rate(
        lambda state: joint_jacobian(mechanism, links, joint, state),
        q,
        q_dot,
    ) @ q_dot
    analytical = joint_acceleration_bias(mechanism, links, joint, q, q_dot)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("kind", ["ground-mobile", "mobile-ground", "mobile-mobile"])
def test_prismatic_joint_acceleration_bias_matches_directional_jacobian_rate(kind):
    mechanism, joint, q, q_dot = _prismatic_case(kind)
    links = mechanism.links
    numerical = _directional_jacobian_rate(
        lambda state: joint_jacobian(mechanism, links, joint, state),
        q,
        q_dot,
    ) @ q_dot
    analytical = joint_acceleration_bias(mechanism, links, joint, q, q_dot)
    np.testing.assert_allclose(analytical, numerical, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("kind", ["ground-mobile", "mobile-ground", "mobile-mobile"])
def test_prismatic_driver_acceleration_bias_matches_directional_jacobian_rate(kind):
    mechanism, joint, q, q_dot = _prismatic_case(kind)
    links = mechanism.links
    numerical = float(
        (_directional_jacobian_rate(
            lambda state: driver_jacobian(mechanism, links, joint, state),
            q,
            q_dot,
        ) @ q_dot)[0]
    )
    analytical = driver_acceleration_bias(mechanism, links, joint, q, q_dot)
    assert analytical == pytest.approx(numerical, rel=3e-6, abs=3e-7)


def test_revolute_driver_acceleration_bias_is_zero():
    mechanism, joint, q, q_dot = _revolute_case("mobile-mobile")
    assert driver_acceleration_bias(mechanism, mechanism.links, joint, q, q_dot) == 0.0
