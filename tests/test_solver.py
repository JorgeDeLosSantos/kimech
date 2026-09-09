from types import SimpleNamespace

import numpy as np
import pytest

from kimech import (
    Configuration,
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


def test_scalar_value_returns_configuration_and_complete_mapping_is_packed_by_link_order():
    mechanism, link, joint = _single_revolute()
    pose = np.array([0.1, -0.1, 0.4])

    config = solve(
        mechanism,
        input=joint,
        values=np.float64(0.5),
        initial_guess={link: pose},
    )

    assert isinstance(config, Configuration)
    assert config.input_joint is joint
    assert config.input_value == pytest.approx(0.5)
    assert config.joint_coordinate(joint) == pytest.approx(0.5)
    np.testing.assert_array_equal(pose, [0.1, -0.1, 0.4])


def test_length_one_sequence_returns_kinematic_solution():
    mechanism, link, joint = _single_revolute()

    solution = solve(
        mechanism,
        input=joint,
        values=[0.5],
        initial_guess={link: (0.0, 0.0, 0.4)},
    )

    assert isinstance(solution, KinematicSolution)
    assert len(solution) == 1
    np.testing.assert_array_equal(solution.input_values, [0.5])


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
            input=joint,
            values=values,
            initial_guess={link: (0.0, 0.0, 0.5)},
        )


def test_solve_rejects_incorrect_argument_types():
    mechanism, link, joint = _single_revolute()

    with pytest.raises(TypeError, match="mechanism"):
        solve(object(), input=joint, values=0.5, initial_guess={link: (0.0, 0.0, 0.5)})
    with pytest.raises(TypeError, match="input"):
        solve(mechanism, input=object(), values=0.5, initial_guess={link: (0.0, 0.0, 0.5)})
    with pytest.raises(TypeError, match="initial_guess"):
        solve(mechanism, input=joint, values=0.5, initial_guess=[0.0, 0.0, 0.5])


def test_initial_guess_mapping_must_contain_exactly_snapshot_links():
    mechanism, link, joint = _single_revolute()
    other, external_link, _ = _single_revolute()

    with pytest.raises(ValueError, match="exactly all links"):
        solve(mechanism, input=joint, values=0.5, initial_guess={})
    with pytest.raises(ValueError, match="exactly all links"):
        solve(
            mechanism,
            input=joint,
            values=0.5,
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
        solve(mechanism, input=joint, values=0.5, initial_guess={link: pose})


def test_configuration_from_same_mechanism_can_be_reused_as_guess():
    mechanism, link, joint = _single_revolute()
    first = Configuration(mechanism, [0.0, 0.0, 0.4])

    second = solve(mechanism, input=joint, values=0.6, initial_guess=first)

    assert second.joint_coordinate(joint) == pytest.approx(0.6)
    assert link.mechanism is mechanism


def test_configuration_from_another_mechanism_is_rejected():
    mechanism, _, joint = _single_revolute()
    other, _, _ = _single_revolute()
    other_config = Configuration(other, [0.0, 0.0, 0.5])

    with pytest.raises(ValueError, match="another mechanism"):
        solve(mechanism, input=joint, values=0.5, initial_guess=other_config)


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
        solve(mechanism, input=input_joint, values=0.5, initial_guess=stale)


def test_disconnected_mechanism_is_rejected_with_validation_errors():
    mechanism, link, joint = _single_revolute()
    mechanism.add_link("orphan")

    with pytest.raises(InvalidModelError, match="disconnected.*orphan"):
        solve(
            mechanism,
            input=joint,
            values=0.5,
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
            input=joint,
            values=0.5,
            initial_guess={link: (0.0, 0.0, 0.5)},
        )


def test_external_input_joint_is_rejected_by_identity():
    mechanism, link, _ = _single_revolute()
    _, _, external_joint = _single_revolute()

    with pytest.raises(InvalidModelError, match="does not belong"):
        solve(
            mechanism,
            input=external_joint,
            values=0.5,
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
            input=joint,
            values=[0.5],
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
            input=joint,
            values=0.5,
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
            input=joint,
            values=[0.5],
            initial_guess={link: (0.0, 0.0, 0.5)},
        )
