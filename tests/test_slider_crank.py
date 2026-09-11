import numpy as np
import pytest

from kimech import Configuration, KinematicSolution, Mechanism, solve
from kimech._constraints import residual


def _slider_crank():
    mechanism = Mechanism("slider_crank")
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


def test_slider_crank_revolute_input_scalar_and_sweep():
    mechanism, crank_joint, _, guess = _slider_crank()

    config = solve(mechanism, input=crank_joint, values=0.7, initial_guess=guess)
    values = np.linspace(0.7, 1.2, 35)
    solution = solve(mechanism, input=crank_joint, values=values, initial_guess=config)

    assert isinstance(config, Configuration)
    assert config.joint_coordinate(crank_joint) == pytest.approx(0.7, abs=1e-10)
    assert _residual_inf(mechanism, crank_joint, config, 0.7) <= 1e-9
    assert isinstance(solution, KinematicSolution)
    np.testing.assert_array_equal(solution.input_values, values)
    np.testing.assert_allclose(solution.joint_coordinates(crank_joint), values, atol=1e-10)
    assert all(
        _residual_inf(mechanism, crank_joint, solution[index], value) <= 1e-9
        for index, value in enumerate(values)
    )


def test_slider_crank_completes_full_revolution_and_returns_to_physical_configuration():
    mechanism, crank_joint, prismatic_joint, guess = _slider_crank()
    values = np.linspace(0.7, 0.7 + 2 * np.pi, 73)

    solution = solve(
        mechanism,
        input=crank_joint,
        values=values,
        initial_guess=guess,
    )

    assert len(solution) == len(values)
    assert values[-1] - values[0] == pytest.approx(2 * np.pi)
    np.testing.assert_allclose(solution.joint_coordinates(crank_joint), values, atol=1e-10)
    assert all(
        _residual_inf(mechanism, crank_joint, solution[index], value) <= 1e-9
        for index, value in enumerate(values)
    )

    slider_positions = solution.joint_coordinates(prismatic_joint)
    assert np.ptp(slider_positions) > 0.0
    assert slider_positions[-1] == pytest.approx(slider_positions[0], abs=1e-8)
    for link in mechanism.links:
        poses = solution.link_poses(link)
        _assert_pose_history_is_continuous(poses)
        np.testing.assert_allclose(poses[-1, :2], poses[0, :2], atol=1e-8)
        assert abs(_wrapped_angle_difference(poses[-1, 2], poses[0, 2])) < 1e-8


def test_slider_crank_prismatic_input_accepts_solver_configuration_as_guess():
    mechanism, crank_joint, prismatic_joint, guess = _slider_crank()
    crank_config = solve(mechanism, input=crank_joint, values=0.7, initial_guess=guess)
    slider_position = crank_config.joint_coordinate(prismatic_joint)

    slider_config = solve(
        mechanism,
        input=prismatic_joint,
        values=slider_position,
        initial_guess=crank_config,
    )

    assert isinstance(slider_config, Configuration)
    assert slider_config.input_joint is prismatic_joint
    assert slider_config.joint_coordinate(prismatic_joint) == pytest.approx(
        slider_position, abs=1e-10
    )
    assert _residual_inf(
        mechanism, prismatic_joint, slider_config, slider_position
    ) <= 1e-9
