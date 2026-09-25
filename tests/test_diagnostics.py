import numpy as np
import pytest

from kimech import Mechanism, KinematicDriver, SolveDiagnostics, solve
from kimech.diagnostics import _jacobian_metrics


def _single_revolute():
    mechanism = Mechanism("single_revolute")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("O", (0.0, 0.0))
    joint = mechanism.revolute(fixed, pivot, name="input")
    return mechanism, link, joint


def _four_bar(scale: float):
    mechanism = Mechanism("four_bar")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30 * scale, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08 * scale, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22 * scale, 0.0))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18 * scale, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)

    guess = {
        crank: (0.0, 0.0, 0.8),
        coupler: (0.05 * scale, 0.06 * scale, 0.2),
        rocker: (0.30 * scale, 0.0, 2.2),
    }
    return mechanism, input_joint, guess


def test_solve_returns_structured_full_rank_diagnostics():
    mechanism, link, joint = _single_revolute()
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.2, 0.4, 0.6],
        ),
        initial_guess={link: (0.0, 0.0, 0.0)},
    )

    diagnostics = solution.diagnostics

    assert isinstance(diagnostics, SolveDiagnostics)
    np.testing.assert_allclose(diagnostics.condition_numbers, np.ones(3))
    np.testing.assert_allclose(diagnostics.min_singular_values, np.ones(3))
    np.testing.assert_array_equal(diagnostics.ranks, np.full(3, 3))
    np.testing.assert_array_equal(
        diagnostics.strategies,
        ["initial_guess", "predictor", "predictor"],
    )
    np.testing.assert_array_equal(diagnostics.corrector_attempts, [1, 1, 1])
    assert np.all(diagnostics.residual_norms <= 1e-9)


def test_diagnostics_follow_solution_slicing():
    mechanism, link, joint = _single_revolute()
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.2, 0.4, 0.6],
        ),
        initial_guess={link: (0.0, 0.0, 0.0)},
    )

    subset = solution[1:]

    assert subset.diagnostics is not None
    np.testing.assert_array_equal(
        subset.diagnostics.condition_numbers,
        solution.diagnostics.condition_numbers[1:],
    )
    np.testing.assert_array_equal(
        subset.diagnostics.ranks,
        solution.diagnostics.ranks[1:],
    )
    np.testing.assert_array_equal(
        subset.diagnostics.subdivision_counts,
        solution.diagnostics.subdivision_counts[1:],
    )
    np.testing.assert_array_equal(
        subset.diagnostics.strategies,
        solution.diagnostics.strategies[1:],
    )
    np.testing.assert_array_equal(
        subset.diagnostics.corrector_attempts,
        solution.diagnostics.corrector_attempts[1:],
    )
    np.testing.assert_array_equal(
        subset.diagnostics.residual_norms,
        solution.diagnostics.residual_norms[1:],
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_scaled_jacobian_diagnostics_are_invariant_to_linear_units(scale):
    mechanism, input_joint, guess = _four_bar(scale)
    values = np.linspace(0.8, 1.2, 9)

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values,
        ),
        initial_guess=guess,
    )

    diagnostics = solution.diagnostics
    assert diagnostics is not None

    reference_mechanism, reference_input, reference_guess = _four_bar(1.0)
    reference = solve(
        reference_mechanism,
        driver=KinematicDriver(
            reference_input,
            position=values,
        ),
        initial_guess=reference_guess,
    ).diagnostics

    np.testing.assert_allclose(
        diagnostics.condition_numbers,
        reference.condition_numbers,
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        diagnostics.min_singular_values,
        reference.min_singular_values,
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_array_equal(diagnostics.ranks, reference.ranks)


def test_manual_solution_may_omit_diagnostics():
    mechanism, _, joint = _single_revolute()
    from kimech import KinematicSolution

    solution = KinematicSolution(
        mechanism,
        KinematicDriver(joint, position=[0.0]),
        [[0.0, 0.0, 0.0]],
    )

    assert solution.diagnostics is None


def test_exactly_singular_scaled_jacobian_reports_infinite_condition_and_rank_loss():
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    condition, minimum, rank = _jacobian_metrics(matrix)

    assert np.isinf(condition)
    assert minimum == pytest.approx(0.0)
    assert rank == 2
