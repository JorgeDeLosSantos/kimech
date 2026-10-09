"""D17: distinguish joint rank loss, dependent drivers and nonconvergence."""

import numpy as np
import pytest

from kimech import KinematicDriver, KinematicSolveError, Mechanism, solve
from kimech._scaling import NumericalScaling
from kimech._differential import _solve_linear_state
from kimech.diagnostics import _rank_analysis


@pytest.mark.parametrize(
    ("matrix", "joint_rows", "expected_issue", "expected_joint_rank", "expected_full_rank"),
    [
        (np.eye(3), 2, "regular", 2, 3),
        (np.array([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]]),
         2, "joint_rank_loss", 1, 2),
        (np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]),
         2, "dependent_drivers", 2, 2),
        (np.diag([1.0, 1.0, 1e-10]), 2, "regular", 2, 3),
    ],
)
def test_rank_analysis_separates_joint_and_driver_independence(
    matrix, joint_rows, expected_issue, expected_joint_rank, expected_full_rank
):
    condition, minimum, rank, joint_rank, issue = _rank_analysis(
        matrix, joint_row_count=joint_rows
    )
    assert rank == expected_full_rank
    assert joint_rank == expected_joint_rank
    assert issue == expected_issue
    assert minimum >= 0.0
    assert condition >= 1.0


@pytest.mark.parametrize(
    ("matrix", "expected_issue"),
    [
        (np.array([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]]),
         "joint_rank_loss"),
        (np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]),
         "dependent_drivers"),
    ],
)
def test_differential_rank_loss_is_reported_without_pseudoinverse(matrix, expected_issue):
    scaling = NumericalScaling(1.0, np.ones(3), np.ones(3))
    with pytest.raises(KinematicSolveError, match="failed to solve velocity") as captured:
        _solve_linear_state(
            matrix, np.array([0.0, 0.0, 0.0]),
            scaling=scaling, link_count=1, joint_row_count=2,
            stage="velocity", driver_value=np.array([0.3, 0.4]), driver_index=3,
        )
    ctx = captured.value.context
    assert ctx.failure_kind == expected_issue
    assert ctx.rank_issue == expected_issue
    assert ctx.joint_rank == (1 if expected_issue == "joint_rank_loss" else 2)
    assert ctx.rank == 2
    assert ctx.sample_index == 3
    assert ctx.driver_positions == pytest.approx((0.3, 0.4))
    assert ctx.driver_position is None


def test_high_condition_number_does_not_automatically_imply_rank_deficiency():
    matrix = np.diag([1.0, 1.0, 1e-10])
    scaling = NumericalScaling(1.0, np.ones(3), np.ones(3))
    velocity = _solve_linear_state(
        matrix, np.array([2., 3., 0.]),
        scaling=scaling, link_count=1, joint_row_count=2,
        stage="velocity", driver_value=0.1, driver_index=0,
    )
    np.testing.assert_allclose(velocity, [2.0, 3.0, 0.0])
    assert _rank_analysis(matrix, joint_row_count=2)[0] >= 1e9


def _toggle_four_bar():
    mechanism = Mechanism()
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (4.0, 0.0))
    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (1.0, 0.0))
    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (2.0, 0.0))
    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (1.0, 0.0))
    input_joint = mechanism.revolute(ground_a, crank_a)
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)
    guess = {crank: (0., 0., 0.), coupler: (1., 0., 0.), rocker: (3., 0., 0.)}
    return mechanism, input_joint, guess


def test_feasible_toggle_position_retained_but_velocity_rank_is_not_unique():
    mechanism, joint, guess = _toggle_four_bar()
    position = solve(
        mechanism, drivers=KinematicDriver(joint, position=0.0), initial_guess=guess
    )
    assert position.diagnostics.ranks[0] < 9
    assert position.diagnostics.joint_ranks[0] < 8
    assert position.diagnostics.rank_issues[0] == "joint_rank_loss"
    summary = position.diagnostics.summary()
    assert summary.minimum_joint_rank == position.diagnostics.joint_ranks[0]
    assert summary.rank_issue_counts == (("joint_rank_loss", 1),)
    assert not position.has_velocity

    with pytest.raises(KinematicSolveError) as captured:
        solve(
            mechanism,
            drivers=KinematicDriver(joint, position=0.0, velocity=1.0),
            initial_guess=guess,
        )
    context = captured.value.context
    assert context.stage == "velocity"
    assert context.failure_kind == "joint_rank_loss"
    assert context.sample_index == 0
    assert context.driver_positions == pytest.approx((0.0,))


def _two_dof_with_dependent_input_joints():
    """One parallelogram four-bar and an independent revolute arm: M=2."""
    mechanism = Mechanism()
    a = mechanism.ground.add_point("A", (0., 0.))
    d = mechanism.ground.add_point("D", (2., 0.))
    g = mechanism.ground.add_point("G", (4., 0.))
    crank = mechanism.add_link("crank")
    c0 = crank.add_point("A", (0., 0.))
    c1 = crank.add_point("B", (1., 0.))
    coupler = mechanism.add_link("coupler")
    b0 = coupler.add_point("B", (0., 0.))
    b1 = coupler.add_point("C", (2., 0.))
    rocker = mechanism.add_link("rocker")
    r0 = rocker.add_point("C", (1., 0.))
    r1 = rocker.add_point("D", (0., 0.))
    arm = mechanism.add_link("independent_arm")
    arm_anchor = arm.add_point("G", (0., 0.))
    jcrank = mechanism.revolute(a, c0)
    mechanism.revolute(c1, b0)
    mechanism.revolute(b1, r0)
    jrocker = mechanism.revolute(d, r1)
    mechanism.revolute(g, arm_anchor)
    guess = {
        crank: (0., 0., np.pi/2),
        coupler: (0., 1., 0.),
        rocker: (2., 0., np.pi/2),
        arm: (4., 0., 0.3),
    }
    return mechanism, jcrank, jrocker, guess


def test_feasible_multi_dof_dependent_drivers_are_distinguished_from_joint_rank_loss():
    mechanism, j1, j2, guess = _two_dof_with_dependent_input_joints()
    drivers = (
        KinematicDriver(j1, position=np.pi / 2),
        KinematicDriver(j2, position=np.pi / 2),
    )
    position = solve(mechanism, drivers=drivers, initial_guess=guess)
    assert position.diagnostics.joint_ranks[0] == 10
    assert position.diagnostics.ranks[0] < 12
    assert position.diagnostics.rank_issues[0] == "dependent_drivers"

    with pytest.raises(KinematicSolveError) as captured:
        solve(
            mechanism,
            drivers=[
                KinematicDriver(j1, position=np.pi/2, velocity=0.5),
                KinematicDriver(j2, position=np.pi/2, velocity=0.5),
            ],
            initial_guess=guess,
        )
    ctx = captured.value.context
    assert ctx.failure_kind == "dependent_drivers"
    assert ctx.joint_rank == 10
    assert ctx.sample_index == 0
    assert len(ctx.driver_positions) == 2


def test_rank_histories_follow_slicing_and_are_defensive_copies():
    mechanism, joint, guess = _toggle_four_bar()
    result = solve(
        mechanism,
        drivers=KinematicDriver(joint, position=[0.0, 0.0]),
        initial_guess=guess,
    )
    subset = result[1:]
    np.testing.assert_array_equal(
        subset.diagnostics.joint_ranks, result.diagnostics.joint_ranks[1:]
    )
    np.testing.assert_array_equal(
        subset.diagnostics.rank_issues, result.diagnostics.rank_issues[1:]
    )
    exposed = result.diagnostics.joint_ranks
    exposed[0] = 100
    assert result.diagnostics.joint_ranks[0] != 100
