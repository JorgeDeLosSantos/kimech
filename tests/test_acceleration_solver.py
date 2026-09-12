import numpy as np
import pytest

from kimech import Configuration, KinematicSolution, KinematicSolveError, Mechanism, solve


def _single_revolute():
    mechanism = Mechanism("single_revolute")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("O", (0.0, 0.0))
    tip = link.add_point("tip", (2.0, 0.0))
    joint = mechanism.revolute(fixed, pivot, name="input")
    guess = {link: (0.0, 0.0, 0.2)}
    return mechanism, link, tip, joint, guess


def _single_prismatic():
    mechanism = Mechanism("single_prismatic")
    guide = mechanism.ground.add_point("G", (0.0, 0.0))
    slider = mechanism.add_link("slider")
    slider_point = slider.add_point("G", (0.0, 0.0))
    joint = mechanism.prismatic(
        guide,
        slider_point,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="input",
    )
    guess = {slider: (0.5, 0.0, 0.0)}
    return mechanism, slider, joint, guess


def test_scalar_revolute_acceleration_matches_prescribed_angular_acceleration():
    mechanism, link, _, joint, guess = _single_revolute()

    config = solve(
        mechanism,
        input=joint,
        values=0.7,
        input_velocity=2.5,
        input_acceleration=-1.2,
        initial_guess=guess,
    )

    assert isinstance(config, Configuration)
    assert config.has_velocity
    assert config.has_acceleration
    assert config.input_acceleration == pytest.approx(-1.2)
    np.testing.assert_allclose(config.body_acceleration(link), [0.0, 0.0, -1.2])
    assert config.joint_acceleration(joint) == pytest.approx(-1.2)


def test_scalar_prismatic_acceleration_matches_prescribed_translation_acceleration():
    mechanism, slider, joint, guess = _single_prismatic()

    config = solve(
        mechanism,
        input=joint,
        values=1.2,
        input_velocity=-0.4,
        input_acceleration=0.75,
        initial_guess=guess,
    )

    np.testing.assert_allclose(config.body_acceleration(slider), [0.75, 0.0, 0.0])
    assert config.joint_acceleration(joint) == pytest.approx(0.75)


def test_point_acceleration_includes_centripetal_term_for_constant_input_speed():
    mechanism, _, tip, joint, guess = _single_revolute()
    theta = 0.4
    omega = 3.0

    config = solve(
        mechanism,
        input=joint,
        values=theta,
        input_velocity=omega,
        input_acceleration=0.0,
        initial_guess=guess,
    )

    radial = np.array([2.0 * np.cos(theta), 2.0 * np.sin(theta)])
    np.testing.assert_allclose(config.acceleration(tip), -(omega**2) * radial, atol=1e-11)


def test_sweep_scalar_acceleration_broadcasts_and_preserves_result_shape():
    mechanism, link, _, joint, guess = _single_revolute()
    values = np.array([0.2, 0.4, 0.6])

    solution = solve(
        mechanism,
        input=joint,
        values=values,
        input_velocity=2.0,
        input_acceleration=-0.5,
        initial_guess=guess,
    )

    assert isinstance(solution, KinematicSolution)
    assert solution.has_acceleration
    np.testing.assert_allclose(solution.input_accelerations, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(solution.body_accelerations(link)[:, 2], [-0.5, -0.5, -0.5])


def test_sweep_accepts_elementwise_acceleration_history():
    mechanism, link, _, joint, guess = _single_revolute()
    values = np.array([0.2, 0.4, 0.6])
    velocities = np.array([1.0, -2.0, 0.0])
    accelerations = np.array([0.5, -0.25, 1.5])

    solution = solve(
        mechanism,
        input=joint,
        values=values,
        input_velocity=velocities,
        input_acceleration=accelerations,
        initial_guess=guess,
    )

    np.testing.assert_array_equal(solution.input_accelerations, accelerations)
    np.testing.assert_allclose(solution.body_accelerations(link)[:, 2], accelerations)
    assert solution[1].input_acceleration == pytest.approx(-0.25)


def test_acceleration_request_does_not_change_position_or_velocity_solution():
    mechanism, _, _, joint, guess = _single_revolute()
    values = np.linspace(0.2, 0.8, 5)

    velocity_only = solve(
        mechanism,
        input=joint,
        values=values,
        input_velocity=4.0,
        initial_guess=guess,
    )
    with_acceleration = solve(
        mechanism,
        input=joint,
        values=values,
        input_velocity=4.0,
        input_acceleration=1.25,
        initial_guess=guess,
    )

    np.testing.assert_array_equal(with_acceleration.coordinates, velocity_only.coordinates)
    np.testing.assert_array_equal(
        with_acceleration.coordinate_velocities,
        velocity_only.coordinate_velocities,
    )


def test_input_acceleration_without_velocity_is_rejected_before_solving():
    mechanism, _, _, joint, guess = _single_revolute()

    with pytest.raises(ValueError, match="requires input_velocity"):
        solve(
            mechanism,
            input=joint,
            values=0.5,
            input_acceleration=1.0,
            initial_guess=guess,
        )


@pytest.mark.parametrize(
    "values, acceleration, message",
    [
        (0.5, [1.0], "scalar"),
        ([0.2, 0.4, 0.6], [1.0], "shape"),
        ([0.2, 0.4, 0.6], [[1.0], [2.0], [3.0]], "1-dimensional"),
        ([0.2, 0.4], [1.0, np.nan], "finite"),
        ([0.2, 0.4], "not-a-number", "numeric"),
    ],
)
def test_input_acceleration_shape_and_finiteness_validation(values, acceleration, message):
    mechanism, _, _, joint, guess = _single_revolute()

    with pytest.raises((TypeError, ValueError), match=message):
        solve(
            mechanism,
            input=joint,
            values=values,
            input_velocity=1.0,
            input_acceleration=acceleration,
            initial_guess=guess,
        )


def test_acceleration_linear_algebra_failure_is_translated(monkeypatch):
    mechanism, _, _, joint, guess = _single_revolute()
    original_solve = np.linalg.solve
    calls = 0

    def fail_second_linear_solve(matrix, rhs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise np.linalg.LinAlgError("deliberate acceleration failure")
        return original_solve(matrix, rhs)

    monkeypatch.setattr("kimech._differential.np.linalg.solve", fail_second_linear_solve)

    with pytest.raises(
        KinematicSolveError,
        match="failed to solve acceleration.*deliberate acceleration failure",
    ):
        solve(
            mechanism,
            input=joint,
            values=0.5,
            input_velocity=1.0,
            input_acceleration=0.0,
            initial_guess=guess,
        )
