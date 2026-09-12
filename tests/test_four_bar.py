import numpy as np
import pytest

from kimech import Configuration, KinematicSolution, Mechanism, solve
from kimech._constraints import residual


def _four_bar():
    mechanism = Mechanism("four_bar")
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


def _residual_inf(mechanism, input_joint, config, value):
    return np.linalg.norm(
        residual(
            mechanism,
            mechanism.links,
            mechanism.joints,
            input_joint,
            config.coordinates,
            value,
        ),
        ord=np.inf,
    )


def _wrapped_angle_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _assert_pose_history_is_continuous(poses):
    positions = poses[:, :2]
    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angle_steps = np.abs(
        np.arctan2(
            np.sin(np.diff(poses[:, 2])),
            np.cos(np.diff(poses[:, 2])),
        )
    )

    assert np.max(angle_steps) < np.deg2rad(30)

    span = np.max(np.ptp(positions, axis=0))
    position_scale = max(1.0, np.max(np.abs(positions)))
    small_tolerance = np.sqrt(np.finfo(poses.dtype).eps) * position_scale
    if span > small_tolerance:
        assert np.max(position_steps) < 0.5 * span
    else:
        assert np.max(position_steps) < small_tolerance


def test_four_bar_scalar_solve_satisfies_complete_constraint_system():
    mechanism, input_joint, _, guess = _four_bar()

    config = solve(mechanism, input=input_joint, values=0.8, initial_guess=guess)

    assert isinstance(config, Configuration)
    assert config.joint_coordinate(input_joint) == pytest.approx(0.8, abs=1e-10)
    assert _residual_inf(mechanism, input_joint, config, 0.8) <= 1e-9


def test_four_bar_sweep_uses_continuation_and_preserves_values():
    mechanism, input_joint, _, guess = _four_bar()
    values = np.linspace(0.8, 1.3, 40)

    solution = solve(mechanism, input=input_joint, values=values, initial_guess=guess)

    assert isinstance(solution, KinematicSolution)
    assert len(solution) == len(values)
    np.testing.assert_array_equal(solution.input_values, values)
    np.testing.assert_allclose(solution.joint_coordinates(input_joint), values, atol=1e-10)
    assert all(
        _residual_inf(mechanism, input_joint, config, value) <= 1e-9
        for config, value in zip((solution[i] for i in range(len(solution))), values)
    )


def test_four_bar_reverse_sweep_preserves_user_direction():
    mechanism, input_joint, _, guess = _four_bar()
    endpoint = solve(mechanism, input=input_joint, values=1.3, initial_guess=guess)
    values = np.linspace(0.8, 1.3, 40)[::-1]

    solution = solve(mechanism, input=input_joint, values=values, initial_guess=endpoint)

    np.testing.assert_array_equal(solution.input_values, values)
    np.testing.assert_allclose(solution.joint_coordinates(input_joint), values, atol=1e-10)


def test_four_bar_completes_full_revolution_and_returns_to_physical_configuration():
    mechanism, input_joint, point_p, guess = _four_bar()
    values = np.linspace(0.8, 0.8 + 2 * np.pi, 73)

    solution = solve(
        mechanism,
        input=input_joint,
        values=values,
        initial_guess=guess,
    )

    assert len(solution) == len(values)
    assert values[-1] - values[0] == pytest.approx(2 * np.pi)
    np.testing.assert_allclose(solution.joint_coordinates(input_joint), values, atol=1e-10)
    assert all(
        _residual_inf(mechanism, input_joint, config, value) <= 1e-9
        for config, value in zip((solution[i] for i in range(len(solution))), values)
    )

    path = solution.point_path(point_p)
    np.testing.assert_allclose(path[-1], path[0], atol=1e-8)
    for link in mechanism.links:
        poses = solution.body_poses(link)
        _assert_pose_history_is_continuous(poses)
        np.testing.assert_allclose(poses[-1, :2], poses[0, :2], atol=1e-8)
        assert abs(_wrapped_angle_difference(poses[-1, 2], poses[0, 2])) < 1e-8
