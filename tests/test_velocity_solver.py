import numpy as np
import pytest

from kimech import Configuration, KinematicSolution, KinematicSolveError, Mechanism, solve


def _single_revolute():
    mechanism = Mechanism("single_revolute")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("O", (0.0, 0.0))
    joint = mechanism.revolute(fixed, pivot, name="input")
    guess = {link: (0.0, 0.0, 0.2)}
    return mechanism, link, joint, guess


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


def test_scalar_revolute_velocity_is_exact_relative_angular_rate():
    mechanism, link, joint, guess = _single_revolute()

    config = solve(
        mechanism,
        input=joint,
        values=0.7,
        input_velocity=2.5,
        initial_guess=guess,
    )

    assert isinstance(config, Configuration)
    assert config.has_velocity
    assert config.input_velocity == pytest.approx(2.5)
    np.testing.assert_allclose(config.body_velocity(link), [0.0, 0.0, 2.5])
    assert config.joint_velocity(joint) == pytest.approx(2.5)


def test_scalar_prismatic_velocity_is_exact_translation_along_axis():
    mechanism, slider, joint, guess = _single_prismatic()

    config = solve(
        mechanism,
        input=joint,
        values=1.2,
        input_velocity=-0.4,
        initial_guess=guess,
    )

    assert config.has_velocity
    np.testing.assert_allclose(config.body_velocity(slider), [-0.4, 0.0, 0.0])
    assert config.joint_velocity(joint) == pytest.approx(-0.4)


def test_sweep_scalar_input_velocity_broadcasts_and_preserves_result_shape():
    mechanism, link, joint, guess = _single_revolute()
    values = np.array([0.2, 0.4, 0.6])

    solution = solve(
        mechanism,
        input=joint,
        values=values,
        input_velocity=3.0,
        initial_guess=guess,
    )

    assert isinstance(solution, KinematicSolution)
    assert solution.has_velocity
    np.testing.assert_allclose(solution.input_velocities, [3.0, 3.0, 3.0])
    np.testing.assert_allclose(
        solution.body_velocities(link),
        [[0.0, 0.0, 3.0], [0.0, 0.0, 3.0], [0.0, 0.0, 3.0]],
    )


def test_sweep_accepts_elementwise_input_velocity_history():
    mechanism, link, joint, guess = _single_revolute()
    values = np.array([0.2, 0.4, 0.6])
    velocities = np.array([1.0, -2.0, 0.0])

    solution = solve(
        mechanism,
        input=joint,
        values=values,
        input_velocity=velocities,
        initial_guess=guess,
    )

    np.testing.assert_array_equal(solution.input_velocities, velocities)
    np.testing.assert_allclose(solution.body_velocities(link)[:, 2], velocities)
    assert solution[1].input_velocity == pytest.approx(-2.0)


def test_zero_input_velocity_requests_and_returns_zero_velocity_state():
    mechanism, link, joint, guess = _single_revolute()

    config = solve(
        mechanism,
        input=joint,
        values=0.5,
        input_velocity=0.0,
        initial_guess=guess,
    )

    assert config.has_velocity
    np.testing.assert_allclose(config.coordinate_velocities, np.zeros(3))
    np.testing.assert_allclose(config.body_velocity(link), np.zeros(3))


def test_velocity_request_does_not_change_position_solution():
    mechanism, _, joint, guess = _single_revolute()
    values = np.linspace(0.2, 0.8, 5)

    position_only = solve(
        mechanism,
        input=joint,
        values=values,
        initial_guess=guess,
    )
    with_velocity = solve(
        mechanism,
        input=joint,
        values=values,
        input_velocity=4.0,
        initial_guess=guess,
    )

    np.testing.assert_array_equal(with_velocity.coordinates, position_only.coordinates)
    assert not position_only.has_velocity
    assert with_velocity.has_velocity


def test_length_one_values_sequence_still_returns_solution_with_scalar_velocity():
    mechanism, _, joint, guess = _single_revolute()

    solution = solve(
        mechanism,
        input=joint,
        values=[0.5],
        input_velocity=2.0,
        initial_guess=guess,
    )

    assert isinstance(solution, KinematicSolution)
    assert solution.coordinate_velocities.shape == (1, 3)
    np.testing.assert_allclose(solution.input_velocities, [2.0])


@pytest.mark.parametrize(
    "values, velocity, message",
    [
        (0.5, [1.0], "scalar"),
        ([0.2, 0.4, 0.6], [1.0], "shape"),
        ([0.2, 0.4, 0.6], [[1.0], [2.0], [3.0]], "1-dimensional"),
        ([0.2, 0.4], [1.0, np.nan], "finite"),
        ([0.2, 0.4], "not-a-number", "numeric"),
    ],
)
def test_input_velocity_shape_and_finiteness_validation(values, velocity, message):
    mechanism, _, joint, guess = _single_revolute()

    with pytest.raises((TypeError, ValueError), match=message):
        solve(
            mechanism,
            input=joint,
            values=values,
            input_velocity=velocity,
            initial_guess=guess,
        )


def test_linear_algebra_failure_is_translated_to_kinematic_solve_error(monkeypatch):
    mechanism, _, joint, guess = _single_revolute()

    def fail_solve(matrix, rhs):
        raise np.linalg.LinAlgError("deliberate singular matrix")

    monkeypatch.setattr("kimech._differential.np.linalg.solve", fail_solve)

    with pytest.raises(
        KinematicSolveError,
        match="failed to solve velocity.*deliberate singular matrix",
    ):
        solve(
            mechanism,
            input=joint,
            values=0.5,
            input_velocity=1.0,
            initial_guess=guess,
        )


def test_inaccurate_linear_solution_is_rejected_by_independent_residual(monkeypatch):
    mechanism, _, joint, guess = _single_revolute()

    def false_solution(matrix, rhs):
        return np.zeros(matrix.shape[1])

    monkeypatch.setattr("kimech._differential.np.linalg.solve", false_solution)

    with pytest.raises(
        KinematicSolveError,
        match=r"failed to solve velocity.*residual_inf=1",
    ):
        solve(
            mechanism,
            input=joint,
            values=0.5,
            input_velocity=1.0,
            initial_guess=guess,
        )


@pytest.mark.parametrize(
    "candidate",
    [np.zeros(2), np.array([0.0, np.nan, 1.0])],
)
def test_malformed_or_nonfinite_linear_candidates_are_rejected(monkeypatch, candidate):
    mechanism, _, joint, guess = _single_revolute()

    def malformed_solution(matrix, rhs):
        return candidate

    monkeypatch.setattr("kimech._differential.np.linalg.solve", malformed_solution)

    with pytest.raises(
        KinematicSolveError,
        match="failed to solve velocity.*residual_inf=unavailable",
    ):
        solve(
            mechanism,
            input=joint,
            values=0.5,
            input_velocity=1.0,
            initial_guess=guess,
        )
