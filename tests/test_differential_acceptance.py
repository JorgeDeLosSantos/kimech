import numpy as np
import pytest

from kimech import KinematicSolveError, Mechanism, solve


def _four_bar():
    mechanism = Mechanism("four_bar_acceptance")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))
    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22, 0.0))
    point_p = coupler.add_point("P", (0.10, 0.05))
    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)
    guess = {
        crank: (0.0, 0.0, 0.8),
        coupler: (0.05, 0.06, 0.2),
        rocker: (0.30, 0.0, 2.2),
    }
    return mechanism, input_joint, point_p, guess


def _slider_crank():
    mechanism = Mechanism("slider_crank_acceptance")
    origin = mechanism.ground.add_point("O", (0.0, 0.0))
    guide = mechanism.ground.add_point("G", (0.0, 0.0))

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))
    rod = mechanism.add_link("connecting_rod")
    rod_b = rod.add_point("B", (0.0, 0.0))
    rod_c = rod.add_point("C", (0.24, 0.0))
    slider = mechanism.add_link("slider")
    slider_c = slider.add_point("C", (0.0, 0.0))
    slider_guide = slider.add_point("G", (0.0, 0.0))

    crank_joint = mechanism.revolute(origin, crank_o, name="crank_input")
    mechanism.revolute(crank_b, rod_b)
    mechanism.revolute(rod_c, slider_c)
    prismatic_joint = mechanism.prismatic(
        guide,
        slider_guide,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="slider_guide",
    )
    guess = {
        crank: (0.0, 0.0, 0.7),
        rod: (0.06, 0.05, -0.2),
        slider: (0.30, 0.0, 0.0),
    }
    return mechanism, crank_joint, prismatic_joint, guess


def test_four_bar_velocity_and_speed_acceleration_scale_with_input_rate():
    mechanism, input_joint, point_p, guess = _four_bar()
    theta = 1.0
    omega = 1.3

    base = solve(
        mechanism,
        input=input_joint,
        values=theta,
        input_velocity=omega,
        input_acceleration=0.0,
        initial_guess=guess,
    )
    doubled = solve(
        mechanism,
        input=input_joint,
        values=theta,
        input_velocity=2.0 * omega,
        input_acceleration=0.0,
        initial_guess=guess,
    )

    np.testing.assert_allclose(doubled.coordinates, base.coordinates, atol=1e-12)
    np.testing.assert_allclose(
        doubled.coordinate_velocities,
        2.0 * base.coordinate_velocities,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        doubled.coordinate_accelerations,
        4.0 * base.coordinate_accelerations,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        doubled.velocity(point_p),
        2.0 * base.velocity(point_p),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        doubled.acceleration(point_p),
        4.0 * base.acceleration(point_p),
        rtol=1e-10,
        atol=1e-11,
    )


def test_four_bar_acceleration_is_linear_in_prescribed_input_acceleration():
    mechanism, input_joint, _, guess = _four_bar()
    theta = 1.0
    alpha = 0.6

    base = solve(
        mechanism,
        input=input_joint,
        values=theta,
        input_velocity=0.0,
        input_acceleration=alpha,
        initial_guess=guess,
    )
    doubled = solve(
        mechanism,
        input=input_joint,
        values=theta,
        input_velocity=0.0,
        input_acceleration=2.0 * alpha,
        initial_guess=guess,
    )

    np.testing.assert_allclose(base.coordinate_velocities, 0.0, atol=1e-14)
    np.testing.assert_allclose(doubled.coordinate_velocities, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        doubled.coordinate_accelerations,
        2.0 * base.coordinate_accelerations,
        rtol=1e-11,
        atol=1e-12,
    )


def test_four_bar_acceleration_splits_into_speed_and_input_acceleration_terms():
    mechanism, input_joint, _, guess = _four_bar()
    theta = 1.0
    omega = 1.3
    alpha = -0.4

    speed_only = solve(
        mechanism,
        input=input_joint,
        values=theta,
        input_velocity=omega,
        input_acceleration=0.0,
        initial_guess=guess,
    )
    alpha_only = solve(
        mechanism,
        input=input_joint,
        values=theta,
        input_velocity=0.0,
        input_acceleration=alpha,
        initial_guess=guess,
    )
    combined = solve(
        mechanism,
        input=input_joint,
        values=theta,
        input_velocity=omega,
        input_acceleration=alpha,
        initial_guess=guess,
    )

    np.testing.assert_allclose(
        combined.coordinate_accelerations,
        speed_only.coordinate_accelerations + alpha_only.coordinate_accelerations,
        rtol=1e-10,
        atol=1e-11,
    )


def test_four_bar_differential_sweep_preserves_input_and_position_history():
    mechanism, input_joint, point_p, guess = _four_bar()
    values = np.linspace(0.8, 1.3, 25)
    velocities = np.linspace(0.5, 1.5, len(values))
    accelerations = np.linspace(-0.3, 0.4, len(values))

    position_only = solve(
        mechanism,
        input=input_joint,
        values=values,
        initial_guess=guess,
    )
    differential = solve(
        mechanism,
        input=input_joint,
        values=values,
        input_velocity=velocities,
        input_acceleration=accelerations,
        initial_guess=guess,
    )

    np.testing.assert_array_equal(differential.coordinates, position_only.coordinates)
    np.testing.assert_array_equal(differential.input_values, values)
    np.testing.assert_array_equal(differential.input_velocities, velocities)
    np.testing.assert_array_equal(differential.input_accelerations, accelerations)
    np.testing.assert_allclose(
        differential.joint_velocities(input_joint), velocities, atol=1e-11
    )
    np.testing.assert_allclose(
        differential.joint_accelerations(input_joint), accelerations, atol=1e-10
    )
    assert differential.point_velocities(point_p).shape == (len(values), 2)
    assert differential.point_accelerations(point_p).shape == (len(values), 2)
    assert np.all(np.isfinite(differential.coordinate_velocities))
    assert np.all(np.isfinite(differential.coordinate_accelerations))


def test_slider_crank_prismatic_inverse_reconstructs_full_differential_state():
    mechanism, crank_joint, prismatic_joint, guess = _slider_crank()
    crank_values = np.linspace(0.72, 0.95, 12)

    forward = solve(
        mechanism,
        input=crank_joint,
        values=crank_values,
        input_velocity=1.1,
        input_acceleration=-0.3,
        initial_guess=guess,
    )
    slider_values = forward.joint_coordinates(prismatic_joint)
    slider_velocities = forward.joint_velocities(prismatic_joint)
    slider_accelerations = forward.joint_accelerations(prismatic_joint)

    inverse = solve(
        mechanism,
        input=prismatic_joint,
        values=slider_values,
        input_velocity=slider_velocities,
        input_acceleration=slider_accelerations,
        initial_guess=forward[0],
    )

    np.testing.assert_allclose(
        inverse.joint_coordinates(crank_joint), crank_values, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(inverse.coordinates, forward.coordinates, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        inverse.coordinate_velocities,
        forward.coordinate_velocities,
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        inverse.coordinate_accelerations,
        forward.coordinate_accelerations,
        rtol=1e-7,
        atol=1e-8,
    )


def test_sweep_acceleration_failure_reports_stage_and_sample_index(monkeypatch):
    mechanism, input_joint, _, guess = _four_bar()
    original_solve = np.linalg.solve
    call_count = 0

    def fail_second_acceleration_solve(matrix, rhs):
        nonlocal call_count
        call_count += 1
        # Three velocity solves occur first, then the acceleration phase begins.
        if call_count == 5:
            raise np.linalg.LinAlgError("deliberate indexed acceleration failure")
        return original_solve(matrix, rhs)

    monkeypatch.setattr("kimech._differential.np.linalg.solve", fail_second_acceleration_solve)

    with pytest.raises(
        KinematicSolveError,
        match=r"failed to solve acceleration at input index 1 .*deliberate indexed acceleration failure",
    ):
        solve(
            mechanism,
            input=input_joint,
            values=[0.9, 1.0, 1.1],
            input_velocity=1.0,
            input_acceleration=0.0,
            initial_guess=guess,
        )
