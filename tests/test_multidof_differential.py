"""Multi-input differential acceptance: analytic 2R and mixed R/P chains."""

import numpy as np
import pytest

from kimech import KinematicDriver, Mechanism, solve


def _unit_direction(theta):
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1)


def _normal_direction(theta):
    return np.stack((-np.sin(theta), np.cos(theta)), axis=-1)


def _serial_2r():
    mechanism = Mechanism()
    ground_point = mechanism.ground.add_point("ground", (0.0, 0.0))
    first = mechanism.add_link("first")
    base = first.add_point("base", (0.0, 0.0))
    elbow = first.add_point("elbow", (2.0, 0.0))
    second = mechanism.add_link("second")
    attachment = second.add_point("attachment", (0.0, 0.0))
    tip = second.add_point("tip", (1.0, 0.0))
    j1 = mechanism.revolute(ground_point, base)
    j2 = mechanism.revolute(elbow, attachment)
    return mechanism, first, second, tip, j1, j2


def test_serial_2r_analytic_positions_velocities_and_accelerations():
    mechanism, first, second, tip, j1, j2 = _serial_2r()
    a = np.array([0.2, 0.35, 0.48])
    b = np.array([0.3, -0.12, 0.19])
    a_dot = np.array([1.1, -0.7, 0.8])
    b_dot = np.array([0.4, 0.5, -1.3])
    a_ddot = np.array([-0.3, 0.7, 0.2])
    b_ddot = np.array([0.6, -0.8, 0.15])
    guess = {
        first: (0.0, 0.0, a[0]),
        second: (2.0 * np.cos(a[0]), 2.0 * np.sin(a[0]), a[0] + b[0]),
    }
    result = solve(
        mechanism,
        drivers=[
            KinematicDriver(j1, position=a, velocity=a_dot, acceleration=a_ddot),
            KinematicDriver(j2, position=b, velocity=b_dot, acceleration=b_ddot),
        ],
        initial_guess=guess,
        time=[0.0, 0.4, 0.9],
    )
    assert result.has_velocity and result.has_acceleration
    q1_vel = np.column_stack((np.zeros(3), np.zeros(3), a_dot))
    q1_acc = np.column_stack((np.zeros(3), np.zeros(3), a_ddot))
    np.testing.assert_allclose(result.body_velocities(first), q1_vel, atol=2e-9)
    np.testing.assert_allclose(result.body_accelerations(first), q1_acc, atol=2e-9)

    tip_v = (2.0 * a_dot[:, None] * _normal_direction(a)
             + (a_dot + b_dot)[:, None] * _normal_direction(a + b))
    tip_a = (2.0 * a_ddot[:, None] * _normal_direction(a)
             - 2.0 * (a_dot ** 2)[:, None] * _unit_direction(a)
             + (a_ddot + b_ddot)[:, None] * _normal_direction(a + b)
             - ((a_dot + b_dot) ** 2)[:, None] * _unit_direction(a + b))
    np.testing.assert_allclose(result.point_velocities(tip), tip_v, atol=2e-9)
    np.testing.assert_allclose(result.point_accelerations(tip), tip_a, atol=2e-9)
    np.testing.assert_allclose(result.joint_velocities(j1), a_dot, atol=2e-9)
    np.testing.assert_allclose(result.joint_velocities(j2), b_dot, atol=2e-9)
    np.testing.assert_allclose(result.joint_accelerations(j1), a_ddot, atol=2e-9)
    np.testing.assert_allclose(result.joint_accelerations(j2), b_ddot, atol=2e-9)

    second_v = result.body_velocities(second)
    np.testing.assert_allclose(second_v[:, 2], a_dot + b_dot, atol=2e-9)
    np.testing.assert_allclose(result.body_accelerations(second)[:, 2],
                               a_ddot + b_ddot, atol=2e-9)

    indexed = result[1]
    assert indexed.has_velocity and indexed.has_acceleration
    assert tuple(d.velocity for d in indexed.drivers) == pytest.approx((a_dot[1], b_dot[1]))
    sliced = result[1:]
    assert sliced.has_velocity and sliced.has_acceleration
    np.testing.assert_allclose(sliced.coordinate_velocities, result.coordinate_velocities[1:])
    np.testing.assert_allclose(sliced.coordinate_accelerations, result.coordinate_accelerations[1:])
    np.testing.assert_allclose(sliced.drivers[1].acceleration, b_ddot[1:])


def test_serial_2r_scalar_differentials_and_velocity_only():
    mechanism, first, second, _, j1, j2 = _serial_2r()
    guess = {first: (0.0, 0.0, 0.2), second: (2.0, 0.0, 0.5)}
    full = solve(
        mechanism, initial_guess=guess,
        drivers=[
            KinematicDriver(j1, position=0.2, velocity=1.2, acceleration=0.6),
            KinematicDriver(j2, position=0.3, velocity=-0.4, acceleration=0.1),
        ],
    )
    assert len(full) == 1
    assert full[0].has_velocity and full[0].has_acceleration
    assert full[0].joint_velocity(j2) == pytest.approx(-0.4)
    assert full[0].joint_acceleration(j1) == pytest.approx(0.6)

    velocity_only = solve(
        mechanism, initial_guess=guess,
        drivers=[
            KinematicDriver(j1, position=[0.2, 0.25], velocity=1.2),
            KinematicDriver(j2, position=[0.3, 0.35], velocity=-0.4),
        ],
    )
    assert velocity_only.has_velocity
    assert not velocity_only.has_acceleration
    np.testing.assert_allclose(velocity_only.joint_velocities(j1), [1.2, 1.2])
    np.testing.assert_allclose(velocity_only.joint_velocities(j2), [-0.4, -0.4])


def test_partial_differential_prescriptions_are_rejected():
    mechanism, first, second, _, j1, j2 = _serial_2r()
    guess = {first: (0.0, 0.0, 0.2), second: (2.0, 0.0, 0.5)}
    with pytest.raises(ValueError, match="all drivers.*velocity"):
        solve(
            mechanism, initial_guess=guess,
            drivers=[
                KinematicDriver(j1, position=0.2, velocity=1.0),
                KinematicDriver(j2, position=0.3),
            ],
        )
    with pytest.raises(ValueError, match="all drivers.*acceleration"):
        solve(
            mechanism, initial_guess=guess,
            drivers=[
                KinematicDriver(j1, position=0.2, velocity=1.0, acceleration=0.0),
                KinematicDriver(j2, position=0.3, velocity=2.0),
            ],
        )
    positions = solve(
        mechanism, initial_guess=guess,
        drivers=[KinematicDriver(j1, position=0.2), KinematicDriver(j2, position=0.3)],
    )
    assert not positions.has_velocity
    assert not positions.has_acceleration


def test_rotating_prismatic_guide_matches_analytic_coriolis_acceleration():
    mechanism = Mechanism()
    ground = mechanism.ground.add_point("O", (0.0, 0.0))
    arm = mechanism.add_link("arm")
    pivot = arm.add_point("O", (0.0, 0.0))
    rail_point = arm.add_point("rail", (1.5, 0.0))
    slider = mechanism.add_link("slider")
    slider_point = slider.add_point("rail", (0.0, 0.0))
    j_angle = mechanism.revolute(ground, pivot)
    j_slide = mechanism.prismatic(
        rail_point, slider_point, axis_a=(1.0, 0.0), axis_b=(1.0, 0.0)
    )
    angle = np.array([0.3, 0.4, 0.55])
    displacement = np.array([0.4, 0.6, 0.8])
    omega = np.array([1.2, 0.7, -0.8])
    speed = np.array([-0.2, 0.9, 0.1])
    alpha = np.array([0.3, -0.4, 0.8])
    acceleration = np.array([0.6, -0.5, -0.3])
    length = 1.5 + displacement
    result = solve(
        mechanism,
        drivers=[
            KinematicDriver(j_angle, position=angle, velocity=omega, acceleration=alpha),
            KinematicDriver(j_slide, position=displacement, velocity=speed,
                            acceleration=acceleration),
        ],
        initial_guess={
            arm: (0.0, 0.0, angle[0]),
            slider: (*((1.5 + displacement[0]) * _unit_direction(angle[0])), angle[0]),
        },
    )
    expected_positions = length[:, None] * _unit_direction(angle)
    expected_velocities = (
        speed[:, None] * _unit_direction(angle)
        + (length * omega)[:, None] * _normal_direction(angle)
    )
    expected_accelerations = (
        (acceleration - length * omega ** 2)[:, None] * _unit_direction(angle)
        + (2.0 * speed * omega + length * alpha)[:, None] * _normal_direction(angle)
    )
    np.testing.assert_allclose(result.body_poses(slider)[:, :2], expected_positions, atol=2e-9)
    np.testing.assert_allclose(result.body_velocities(slider)[:, :2], expected_velocities, atol=2e-9)
    np.testing.assert_allclose(result.body_accelerations(slider)[:, :2], expected_accelerations, atol=2e-9)
    np.testing.assert_allclose(result.joint_velocities(j_slide), speed, atol=2e-9)
    np.testing.assert_allclose(result.joint_accelerations(j_slide), acceleration, atol=2e-9)
