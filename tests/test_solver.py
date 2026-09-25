from types import SimpleNamespace

import numpy as np
import pytest

from kimech import (
    Configuration,
    KinematicDriver,
    InvalidModelError,
    KinematicSolution,
    KinematicSolveError,
    Mechanism,
    solve,
)


def _single_revolute():
    mechanism = Mechanism("single_revolute")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("O", (0.0, 0.0))
    joint = mechanism.revolute(fixed, pivot, name="input")
    return mechanism, link, joint


def test_scalar_position_returns_length_one_solution_and_complete_mapping_is_packed_by_link_order():
    mechanism, link, joint = _single_revolute()
    pose = np.array([0.1, -0.1, 0.4])

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=np.float64(0.5),
        ),
        initial_guess={link: pose},
    )
    config = solution[0]

    assert isinstance(solution, KinematicSolution)
    assert len(solution) == 1
    assert isinstance(config, Configuration)
    assert config.input_joint is joint
    assert config.input_position == pytest.approx(0.5)
    assert config.joint_coordinate(joint) == pytest.approx(0.5)
    np.testing.assert_array_equal(pose, [0.1, -0.1, 0.4])


def test_length_one_sequence_returns_kinematic_solution():
    mechanism, link, joint = _single_revolute()

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.5],
        ),
        initial_guess={link: (0.0, 0.0, 0.4)},
    )

    assert isinstance(solution, KinematicSolution)
    assert len(solution) == 1
    np.testing.assert_array_equal(solution.input_positions, [0.5])


@pytest.mark.parametrize(
    "values, message",
    [
        ([], "empty"),
        ([[0.5]], "1-dimensional"),
        ([np.nan], "finite"),
        ([np.inf], "finite"),
        (["not-a-number"], "numeric"),
    ],
)
def test_invalid_values_are_rejected(values, message):
    mechanism, link, joint = _single_revolute()

    with pytest.raises((TypeError, ValueError), match=message):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=values,
            ),
            initial_guess={link: (0.0, 0.0, 0.5)},
        )


def test_solve_rejects_incorrect_argument_types():
    mechanism, link, joint = _single_revolute()

    with pytest.raises(TypeError, match="mechanism"):
        solve(object(), driver=KinematicDriver(joint, position=0.5), initial_guess={link: (0.0, 0.0, 0.5)})
    with pytest.raises(TypeError, match="joint"):
        KinematicDriver(object(), position=0.5)
    with pytest.raises(TypeError, match="driver"):
        solve(mechanism, driver=object(), initial_guess={link: (0.0, 0.0, 0.5)})
    with pytest.raises(TypeError, match="initial_guess"):
        solve(mechanism, driver=KinematicDriver(joint, position=0.5), initial_guess=[0.0, 0.0, 0.5])


def test_initial_guess_mapping_must_contain_exactly_snapshot_links():
    mechanism, link, joint = _single_revolute()
    other, external_link, _ = _single_revolute()

    with pytest.raises(ValueError, match="exactly all links"):
        solve(mechanism, driver=KinematicDriver(joint, position=0.5), initial_guess={})
    with pytest.raises(ValueError, match="exactly all links"):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=0.5,
            ),
            initial_guess={link: (0.0, 0.0, 0.5), external_link: (0.0, 0.0, 0.5)},
        )

    assert other is external_link.mechanism


@pytest.mark.parametrize(
    "pose, message",
    [
        ((0.0, 0.5), "shape"),
        ((0.0, np.nan, 0.5), "finite"),
    ],
)
def test_initial_guess_poses_must_have_valid_shape_and_finite_values(pose, message):
    mechanism, link, joint = _single_revolute()

    with pytest.raises(ValueError, match=message):
        solve(mechanism, driver=KinematicDriver(joint, position=0.5), initial_guess={link: pose})


def test_configuration_from_same_mechanism_can_be_reused_as_guess():
    mechanism, link, joint = _single_revolute()
    first = Configuration(mechanism, [0.0, 0.0, 0.4])

    second = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=0.6,
        ),
        initial_guess=first,
    )[0]

    assert second.joint_coordinate(joint) == pytest.approx(0.6)
    assert link.mechanism is mechanism


def test_configuration_from_another_mechanism_is_rejected():
    mechanism, _, joint = _single_revolute()
    other, _, _ = _single_revolute()
    other_config = Configuration(other, [0.0, 0.0, 0.5])

    with pytest.raises(ValueError, match="another mechanism"):
        solve(mechanism, driver=KinematicDriver(joint, position=0.5), initial_guess=other_config)


def test_stale_configuration_is_rejected_even_when_modified_model_still_has_mobility_one():
    mechanism, first, input_joint = _single_revolute()
    stale = Configuration(mechanism, [0.0, 0.0, 0.5])
    first_a = first.add_point("A", (1.0, 0.0))

    second = mechanism.add_link("second")
    second_a = second.add_point("A", (0.0, 0.0))
    second_b = second.add_point("B", (1.0, 0.0))
    third = mechanism.add_link("third")
    third_b = third.add_point("B", (0.0, 0.0))
    third_c = third.add_point("C", (1.0, 0.0))
    fixed_c = mechanism.ground.add_point("C", (1.0, 0.0))
    mechanism.revolute(first_a, second_a)
    mechanism.revolute(second_b, third_b)
    mechanism.revolute(third_c, fixed_c)

    assert mechanism.validate().is_valid
    assert mechanism.mobility() == 1
    with pytest.raises(ValueError, match="incompatible"):
        solve(mechanism, driver=KinematicDriver(input_joint, position=0.5), initial_guess=stale)


def test_disconnected_mechanism_is_rejected_with_validation_errors():
    mechanism, link, joint = _single_revolute()
    mechanism.add_link("orphan")

    with pytest.raises(InvalidModelError, match="disconnected.*orphan"):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=0.5,
            ),
            initial_guess={link: (0.0, 0.0, 0.5)},
        )


def test_valid_mechanism_with_non_unit_mobility_is_rejected():
    mechanism, link, joint = _single_revolute()
    fixed = mechanism.ground.add_point("P", (1.0, 0.0))
    point = link.add_point("P", (1.0, 0.0))
    mechanism.revolute(fixed, point)

    assert mechanism.validate().is_valid
    with pytest.raises(InvalidModelError, match="mobility 1"):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=0.5,
            ),
            initial_guess={link: (0.0, 0.0, 0.5)},
        )


def test_external_input_joint_is_rejected_by_identity():
    mechanism, link, _ = _single_revolute()
    _, _, external_joint = _single_revolute()

    with pytest.raises(InvalidModelError, match="does not belong"):
        solve(
            mechanism,
            driver=KinematicDriver(
                external_joint,
                position=0.5,
            ),
            initial_guess={link: (0.0, 0.0, 0.5)},
        )


def test_solver_failure_raises_kinematic_solve_error(monkeypatch):
    mechanism, link, joint = _single_revolute()

    def failed_root(fun, x0, *, jac, method):
        assert callable(fun)
        assert callable(jac)
        assert method == "hybr"
        return SimpleNamespace(success=False, message="deliberate failure", x=x0)

    monkeypatch.setattr("kimech.solver.optimize.root", failed_root)

    with pytest.raises(KinematicSolveError, match="deliberate failure"):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=[0.5],
            ),
            initial_guess={link: (0.0, 0.0, 0.0)},
        )


def test_solver_success_with_bad_independently_recomputed_residual_is_rejected(monkeypatch):
    mechanism, link, joint = _single_revolute()

    def false_success(fun, x0, *, jac, method):
        np.testing.assert_equal(jac(x0).shape, (3, 3))
        assert method == "hybr"
        return SimpleNamespace(success=True, message="claimed success", x=np.zeros(3))

    monkeypatch.setattr("kimech.solver.optimize.root", false_success)

    with pytest.raises(KinematicSolveError, match=r"residual_inf=0\.5"):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=0.5,
            ),
            initial_guess={link: (0.0, 0.0, 0.0)},
        )


@pytest.mark.parametrize("candidate", [np.zeros(2), np.array([0.0, np.nan, 0.5])])
def test_solver_rejects_malformed_or_nonfinite_candidates(monkeypatch, candidate):
    mechanism, link, joint = _single_revolute()

    def malformed_success(fun, x0, *, jac, method):
        return SimpleNamespace(success=True, message="claimed success", x=candidate)

    monkeypatch.setattr("kimech.solver.optimize.root", malformed_success)

    with pytest.raises(KinematicSolveError, match="residual_inf=unavailable"):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=[0.5],
            ),
            initial_guess={link: (0.0, 0.0, 0.5)},
        )


def test_position_sweep_uses_input_tangent_predictor(monkeypatch):
    mechanism, link, joint = _single_revolute()
    guesses = []

    def fake_solve_configuration(
        mechanism_arg,
        links,
        joints,
        input_joint,
        input_value,
        initial_q,
        scaling,
        *,
        input_index=None,
    ):
        assert mechanism_arg is mechanism
        guesses.append(initial_q.copy())
        return np.array([0.0, 0.0, input_value])

    def fake_input_tangent(
        mechanism_arg,
        links,
        joints,
        input_joint,
        q,
        input_value,
        scaling,
        *,
        input_index=None,
    ):
        assert mechanism_arg is mechanism
        return np.array([0.0, 0.0, 1.0])

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_solve_configuration)
    monkeypatch.setattr("kimech.solver.solve_input_tangent", fake_input_tangent)

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.5, 0.7],
        ),
        initial_guess={link: (0.0, 0.0, 0.0)},
    )

    assert len(solution) == 2
    np.testing.assert_allclose(guesses[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(guesses[1], [0.0, 0.0, 0.7])
    np.testing.assert_array_equal(
        solution.diagnostics.strategies,
        ["initial_guess", "predictor"],
    )
    np.testing.assert_array_equal(
        solution.diagnostics.corrector_attempts,
        [1, 1],
    )


def test_position_sweep_retries_warm_start_when_predictor_corrector_fails(monkeypatch):
    mechanism, link, joint = _single_revolute()
    guesses = []

    def fake_solve_configuration(
        mechanism_arg,
        links,
        joints,
        input_joint,
        input_value,
        initial_q,
        scaling,
        *,
        input_index=None,
    ):
        guesses.append(initial_q.copy())
        if input_index == 1 and initial_q[2] == pytest.approx(0.7):
            raise KinematicSolveError("deliberate predictor failure")
        return np.array([0.0, 0.0, input_value])

    def fake_input_tangent(*args, **kwargs):
        return np.array([0.0, 0.0, 1.0])

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_solve_configuration)
    monkeypatch.setattr("kimech.solver.solve_input_tangent", fake_input_tangent)

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.5, 0.7],
        ),
        initial_guess={link: (0.0, 0.0, 0.0)},
    )

    assert len(solution) == 2
    assert len(guesses) == 3
    np.testing.assert_allclose(guesses[1], [0.0, 0.0, 0.7])
    np.testing.assert_allclose(guesses[2], [0.0, 0.0, 0.5])
    assert solution.diagnostics.strategies[1] == "warm_start"
    assert solution.diagnostics.corrector_attempts[1] == 2


def test_position_sweep_falls_back_to_warm_start_when_tangent_solve_fails(monkeypatch):
    mechanism, link, joint = _single_revolute()
    guesses = []

    def fake_solve_configuration(
        mechanism_arg,
        links,
        joints,
        input_joint,
        input_value,
        initial_q,
        scaling,
        *,
        input_index=None,
    ):
        guesses.append(initial_q.copy())
        return np.array([0.0, 0.0, input_value])

    def failed_input_tangent(*args, **kwargs):
        raise KinematicSolveError("deliberate tangent failure")

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_solve_configuration)
    monkeypatch.setattr("kimech.solver.solve_input_tangent", failed_input_tangent)

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.5, 0.7],
        ),
        initial_guess={link: (0.0, 0.0, 0.0)},
    )

    assert len(solution) == 2
    np.testing.assert_allclose(guesses[1], [0.0, 0.0, 0.5])
    assert solution.diagnostics.strategies[1] == "warm_start"
    assert solution.diagnostics.corrector_attempts[1] == 1


def test_adaptive_subdivision_recovers_failed_requested_step(monkeypatch):
    mechanism, link, joint = _single_revolute()
    calls = []

    def fake_solve_configuration(
        mechanism_arg,
        links,
        joints,
        input_joint,
        input_value,
        initial_q,
        scaling,
        *,
        input_index=None,
    ):
        calls.append(float(input_value))
        current = float(initial_q[2])
        if abs(input_value - current) > 0.11:
            raise KinematicSolveError("step too large")
        return np.array([0.0, 0.0, input_value])

    def fake_predict(*args, **kwargs):
        return args[4].copy()

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_solve_configuration)
    monkeypatch.setattr("kimech.solver._predict_next_configuration", fake_predict)

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.0, 0.2],
        ),
        initial_guess={link: (0.0, 0.0, 0.0)},
    )

    np.testing.assert_array_equal(solution.input_positions, [0.0, 0.2])
    np.testing.assert_allclose(solution.coordinates[:, 2], [0.0, 0.2])
    assert solution.diagnostics is not None
    np.testing.assert_array_equal(solution.diagnostics.subdivision_counts, [0, 1])
    assert solution.diagnostics.strategies[1] == "subdivision"
    assert solution.diagnostics.corrector_attempts[1] == 3
    assert 0.1 in calls


def test_adaptive_subdivision_hides_internal_samples(monkeypatch):
    mechanism, link, joint = _single_revolute()

    def fake_solve_configuration(
        mechanism_arg,
        links,
        joints,
        input_joint,
        input_value,
        initial_q,
        scaling,
        *,
        input_index=None,
    ):
        current = float(initial_q[2])
        if abs(input_value - current) > 0.06:
            raise KinematicSolveError("step too large")
        return np.array([0.0, 0.0, input_value])

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_solve_configuration)
    monkeypatch.setattr(
        "kimech.solver._predict_next_configuration",
        lambda mechanism_arg, links, joints, input_joint, q, input_value, next_input_value, scaling, input_index=None:
            q.copy(),
    )

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.0, 0.2],
        ),
        initial_guess={link: (0.0, 0.0, 0.0)},
    )

    assert len(solution) == 2
    np.testing.assert_array_equal(solution.input_positions, [0.0, 0.2])
    assert solution.diagnostics.subdivision_counts[1] == 3
    assert solution.diagnostics.strategies[1] == "subdivision"
    assert solution.diagnostics.corrector_attempts[1] == 7


def test_impossible_target_preserves_original_failure_after_subdivision(monkeypatch):
    mechanism, link, joint = _single_revolute()

    def fake_solve_configuration(
        mechanism_arg,
        links,
        joints,
        input_joint,
        input_value,
        initial_q,
        scaling,
        *,
        input_index=None,
    ):
        if input_value > 0.1:
            raise KinematicSolveError(f"unreachable target {input_value}")
        return np.array([0.0, 0.0, input_value])

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_solve_configuration)
    monkeypatch.setattr(
        "kimech.solver._predict_next_configuration",
        lambda mechanism_arg, links, joints, input_joint, q, input_value, next_input_value, scaling, input_index=None:
            q.copy(),
    )

    with pytest.raises(KinematicSolveError, match=r"unreachable target 0\.2"):
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=[0.0, 0.2],
            ),
            initial_guess={link: (0.0, 0.0, 0.0)},
        )
