import numpy as np
import pytest

from kimech import Configuration, Mechanism, RevoluteJoint
from kimech._constraints import (
    driver_jacobian,
    driver_residual,
    jacobian,
    joint_jacobian,
    joint_residual,
    residual,
)
from kimech._geometry import rotation_matrix


def finite_difference_jacobian(func, q, eps=1e-7):
    q = np.asarray(q, dtype=float)
    result = np.empty((func(q).size, q.size), dtype=float)
    for column in range(q.size):
        step = eps * max(1.0, abs(q[column]))
        offset = np.zeros_like(q)
        offset[column] = step
        result[:, column] = (func(q + offset) - func(q - offset)) / (2.0 * step)
    return result


def assert_jacobian_matches(function, analytical, q):
    np.testing.assert_allclose(
        analytical,
        finite_difference_jacobian(function, q),
        rtol=1e-6,
        atol=1e-7,
    )


def _revolute_case(kind):
    mechanism = Mechanism()
    mobile_a = mechanism.add_link("a")
    point_a = mobile_a.add_point("A", (0.7, -0.4))
    mobile_b = mechanism.add_link("b")
    point_b = mobile_b.add_point("B", (-0.3, 0.8))
    ground_a = mechanism.ground.add_point("GA", (1.2, -0.6))
    ground_b = mechanism.ground.add_point("GB", (-0.5, 0.9))
    if kind == "ground-mobile":
        joint = mechanism.revolute(ground_a, point_b)
    elif kind == "mobile-ground":
        joint = mechanism.revolute(point_a, ground_b)
    else:
        joint = mechanism.revolute(point_a, point_b)
    q = np.array([0.4, -1.1, 0.37, 1.5, 0.2, -0.61])
    return mechanism, joint, q


def _prismatic_case(kind):
    mechanism = Mechanism()
    body_a = mechanism.add_link("a")
    point_a = body_a.add_point("A", (0.6, -0.35))
    body_b = mechanism.add_link("b")
    point_b = body_b.add_point("B", (-0.45, 0.7))
    ground = mechanism.ground.add_point("G", (0.2, -0.8))
    if kind == "horizontal":
        joint = mechanism.prismatic(ground, point_b, axis_a=(1, 0), axis_b=(1, 0))
    elif kind == "inclined":
        joint = mechanism.prismatic(
            ground, point_b, axis_a=(0.6, 0.8), axis_b=(-0.2, 0.98)
        )
    else:
        joint = mechanism.prismatic(
            point_a, point_b, axis_a=(0.6, 0.8), axis_b=(0.9, -0.3)
        )
    q = np.array([0.3, -0.9, 0.43, 1.7, 0.6, -0.52])
    return mechanism, joint, q


def test_revolute_known_ground_mobile_and_mobile_mobile_residuals():
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("fixed", (2.0, -1.0))
    first = mechanism.add_link("first")
    first_point = first.add_point("P", (0.4, -0.7))
    ground_joint = mechanism.revolute(fixed, first_point)
    theta_first = 0.63
    first_position = fixed.local - rotation_matrix(theta_first) @ first_point.local

    second = mechanism.add_link("second")
    second_point = second.add_point("Q", (-0.8, 0.35))
    mobile_joint = mechanism.revolute(first_point, second_point)
    theta_second = -0.41
    common = first_position + rotation_matrix(theta_first) @ first_point.local
    second_position = common - rotation_matrix(theta_second) @ second_point.local
    q = np.r_[first_position, theta_first, second_position, theta_second]

    np.testing.assert_allclose(
        joint_residual(mechanism, mechanism.links, ground_joint, q),
        [0.0, 0.0],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        joint_residual(mechanism, mechanism.links, mobile_joint, q),
        [0.0, 0.0],
        atol=1e-15,
    )


def test_prismatic_known_horizontal_and_inclined_mobile_mobile_residuals():
    horizontal = Mechanism()
    ground_point = horizontal.ground.add_point("G", (1.0, 2.0))
    slider = horizontal.add_link("slider")
    slider_point = slider.add_point("S", (-0.4, 0.3))
    horizontal_joint = horizontal.prismatic(
        ground_point, slider_point, axis_a=(1, 0), axis_b=(1, 0)
    )
    q_horizontal = np.array([4.4, 1.7, 0.0])
    np.testing.assert_allclose(
        joint_residual(horizontal, horizontal.links, horizontal_joint, q_horizontal),
        [0.0, 0.0],
        atol=1e-15,
    )

    inclined = Mechanism()
    body_a = inclined.add_link("a")
    point_a = body_a.add_point("A", (0.6, -0.2))
    body_b = inclined.add_link("b")
    point_b = body_b.add_point("B", (-0.3, 0.5))
    alpha_a = 0.37
    alpha_b = -0.21
    inclined_joint = inclined.prismatic(
        point_a,
        point_b,
        axis_a=(np.cos(alpha_a), np.sin(alpha_a)),
        axis_b=(np.cos(alpha_b), np.sin(alpha_b)),
    )
    position_a = np.array([0.8, -1.2])
    theta_a = 0.48
    theta_b = theta_a + alpha_a - alpha_b
    global_point_a = position_a + rotation_matrix(theta_a) @ point_a.local
    global_axis = rotation_matrix(theta_a) @ np.asarray(inclined_joint.axis_a)
    global_point_b = global_point_a + 2.3 * global_axis
    position_b = global_point_b - rotation_matrix(theta_b) @ point_b.local
    q_inclined = np.r_[position_a, theta_a, position_b, theta_b]
    np.testing.assert_allclose(
        joint_residual(inclined, inclined.links, inclined_joint, q_inclined),
        [0.0, 0.0],
        atol=1e-14,
    )


def test_prismatic_relative_axis_angle_is_continuous_across_atan2_branch_cut():
    mechanism = Mechanism()
    body_a = mechanism.add_link("a")
    point_a = body_a.add_point("A", (0.4, -0.3))
    body_b = mechanism.add_link("b")
    point_b = body_b.add_point("B", (-0.2, 0.6))
    alpha_a = np.deg2rad(179.0)
    alpha_b = np.deg2rad(-179.0)
    axis_a = (np.cos(alpha_a), np.sin(alpha_a))
    axis_b = (np.cos(alpha_b), np.sin(alpha_b))
    joint = mechanism.prismatic(
        point_a, point_b, axis_a=axis_a, axis_b=axis_b
    )

    position_a = np.array([0.7, -1.1])
    theta_a = 0.43
    theta_b = theta_a - np.deg2rad(2.0)
    global_point_a = position_a + rotation_matrix(theta_a) @ point_a.local
    global_axis = rotation_matrix(theta_a) @ np.asarray(joint.axis_a)
    global_point_b = global_point_a + 1.8 * global_axis
    position_b = global_point_b - rotation_matrix(theta_b) @ point_b.local
    q = np.r_[position_a, theta_a, position_b, theta_b]

    np.testing.assert_allclose(
        joint_residual(mechanism, mechanism.links, joint, q),
        [0.0, 0.0],
        atol=1e-14,
    )


def test_prismatic_mobile_ground_residual_and_jacobians():
    mechanism = Mechanism()
    mobile = mechanism.add_link("mobile")
    mobile_point = mobile.add_point("M", (0.65, -0.4))
    ground_point = mechanism.ground.add_point("G", (-0.3, 1.2))
    joint = mechanism.prismatic(
        mobile_point,
        ground_point,
        axis_a=(0.6, 0.8),
        axis_b=(-0.45, 0.89),
    )
    links = mechanism.links
    q = np.array([0.75, -1.1, 0.47])

    phi = joint_residual(mechanism, links, joint, q)
    assert phi.shape == (2,)
    assert np.all(np.isfinite(phi))
    assert_jacobian_matches(
        lambda value: joint_residual(mechanism, links, joint, value),
        joint_jacobian(mechanism, links, joint, q),
        q,
    )

    input_value = -0.28
    assert_jacobian_matches(
        lambda value: driver_residual(mechanism, links, joint, value, input_value),
        driver_jacobian(mechanism, links, joint, q),
        q,
    )


@pytest.mark.parametrize("kind", ["ground-mobile", "mobile-ground", "mobile-mobile"])
def test_revolute_joint_jacobian_matches_finite_differences(kind):
    mechanism, joint, q = _revolute_case(kind)
    links = mechanism.links
    assert_jacobian_matches(
        lambda value: joint_residual(mechanism, links, joint, value),
        joint_jacobian(mechanism, links, joint, q),
        q,
    )


@pytest.mark.parametrize("kind", ["horizontal", "inclined", "mobile-mobile"])
def test_prismatic_joint_jacobian_matches_finite_differences(kind):
    mechanism, joint, q = _prismatic_case(kind)
    links = mechanism.links
    assert_jacobian_matches(
        lambda value: joint_residual(mechanism, links, joint, value),
        joint_jacobian(mechanism, links, joint, q),
        q,
    )


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_driver_coordinate_matches_configuration_and_jacobian(kind):
    if kind == "revolute":
        mechanism, joint, q = _revolute_case("mobile-mobile")
    else:
        mechanism, joint, q = _prismatic_case("mobile-mobile")
    links = mechanism.links
    input_value = -0.34
    expected_coordinate = Configuration(mechanism, q).joint_coordinate(joint)
    np.testing.assert_allclose(
        driver_residual(mechanism, links, joint, q, input_value),
        [expected_coordinate - input_value],
    )
    assert_jacobian_matches(
        lambda value: driver_residual(mechanism, links, joint, value, input_value),
        driver_jacobian(mechanism, links, joint, q),
        q,
    )


def _four_bar():
    mechanism = Mechanism("four_bar")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (3.8, 0.2))
    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.2, -0.1))
    crank_b = crank.add_point("B", (1.1, 0.15))
    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (-0.25, 0.3))
    coupler_c = coupler.add_point("C", (1.8, -0.2))
    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.1, -0.35))
    rocker_d = rocker.add_point("D", (1.0, 0.25))
    input_joint = mechanism.revolute(ground_a, crank_a)
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)
    return mechanism, input_joint


def _slider_crank():
    mechanism = Mechanism("slider_crank")
    origin = mechanism.ground.add_point("O", (0.0, 0.0))
    guide = mechanism.ground.add_point("G", (0.1, -0.2))
    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.15, -0.1))
    crank_a = crank.add_point("A", (1.0, 0.2))
    rod = mechanism.add_link("rod")
    rod_a = rod.add_point("A", (-0.3, 0.25))
    rod_b = rod.add_point("B", (1.6, -0.15))
    slider = mechanism.add_link("slider")
    slider_b = slider.add_point("B", (-0.2, 0.35))
    slider_guide = slider.add_point("G", (0.4, -0.25))
    input_joint = mechanism.revolute(origin, crank_o)
    mechanism.revolute(crank_a, rod_a)
    mechanism.revolute(rod_b, slider_b)
    mechanism.prismatic(guide, slider_guide, axis_a=(1, 0), axis_b=(1, 0))
    return mechanism, input_joint


@pytest.mark.parametrize("builder", [_four_bar, _slider_crank])
def test_complete_nine_equation_system_jacobian_matches_finite_differences(builder):
    mechanism, input_joint = builder()
    links = mechanism.links
    joints = mechanism.joints
    q = np.array([0.2, -0.15, 0.38, 1.25, 0.45, -0.27, 3.0, 0.12, 0.19])
    input_value = 0.31
    phi = residual(mechanism, links, joints, input_joint, q, input_value)
    analytical = jacobian(mechanism, links, joints, input_joint, q, input_value)
    assert phi.shape == (9,)
    assert analytical.shape == (9, 9)
    assert_jacobian_matches(
        lambda value: residual(
            mechanism, links, joints, input_joint, value, input_value
        ),
        analytical,
        q,
    )


def test_assembly_preserves_explicit_link_and_joint_snapshot_order():
    mechanism, input_joint = _four_bar()
    links = tuple(reversed(mechanism.links))
    joints = tuple(reversed(mechanism.joints))
    q = np.array([2.9, 0.1, 0.2, 1.3, 0.4, -0.3, 0.2, -0.1, 0.4])
    phi = residual(mechanism, links, joints, input_joint, q, 0.1)
    for index, joint in enumerate(joints):
        np.testing.assert_allclose(
            phi[2 * index : 2 * index + 2],
            joint_residual(mechanism, links, joint, q),
        )
    np.testing.assert_allclose(
        phi[-1:], driver_residual(mechanism, links, input_joint, q, 0.1)
    )


def test_constraints_validate_inputs_and_snapshot_membership():
    mechanism, input_joint = _four_bar()
    links = mechanism.links
    joints = mechanism.joints
    q = np.zeros(9)
    with pytest.raises(ValueError, match="shape"):
        residual(mechanism, links, joints, input_joint, q[:-1], 0.0)
    with pytest.raises(ValueError, match="finite"):
        residual(mechanism, links, joints, input_joint, q * np.nan, 0.0)
    with pytest.raises(ValueError, match="scalar"):
        residual(mechanism, links, joints, input_joint, q, [0.0])
    with pytest.raises(ValueError, match="finite"):
        residual(mechanism, links, joints, input_joint, q, np.inf)

    other, other_input = _four_bar()
    with pytest.raises(ValueError, match="links snapshot"):
        joint_residual(mechanism, links, other_input, q)
    with pytest.raises(ValueError, match="input_joint"):
        residual(mechanism, links, joints, other_input, q, 0.0)

    unregistered = RevoluteJoint(input_joint.point_a, input_joint.point_b)
    with pytest.raises(ValueError, match="does not belong"):
        residual(mechanism, links, (unregistered, *joints[1:]), unregistered, q, 0.0)
    assert other is other_input.point_a.body.mechanism