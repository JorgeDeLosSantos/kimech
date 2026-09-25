from types import SimpleNamespace

import numpy as np
import pytest

from kimech import (
    KinematicDriver,    KinematicSolveError,
    Mechanism,
    SolveDiagnosticSummary,
    SolveDiagnostics,
    SolveFailureContext,
    solve,
)


def _single_revolute():
    mechanism = Mechanism("single_revolute")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("O", (0.0, 0.0))
    joint = mechanism.revolute(fixed, pivot, name="input")
    return mechanism, link, joint


def test_kinematic_solve_error_remains_compatible_without_context():
    error = KinematicSolveError("legacy message")

    assert str(error) == "legacy message"
    assert error.context is None


def test_solve_failure_context_is_immutable_and_validated():
    context = SolveFailureContext(
        stage="position",
        input_index=2,
        input_position=0.7,
        residual_norm=1e-5,
        condition_number=12.0,
        min_singular_value=0.08,
        rank=3,
        attempted_strategies=("predictor", "warm_start"),
        corrector_attempts=2,
    )

    assert context.stage == "position"
    assert context.input_index == 2
    assert context.attempted_strategies == ("predictor", "warm_start")

    with pytest.raises(AttributeError):
        context.stage = "velocity"


def test_position_failure_exposes_structured_context(monkeypatch):
    mechanism, link, joint = _single_revolute()

    def failed_root(fun, x0, *, jac, method):
        return SimpleNamespace(
            success=False,
            message="deliberate failure",
            x=np.asarray(x0, dtype=float),
        )

    monkeypatch.setattr("kimech.solver.optimize.root", failed_root)

    with pytest.raises(KinematicSolveError) as captured:
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=[0.5],
            ),
            initial_guess={link: (0.0, 0.0, 0.0)},
        )

    context = captured.value.context
    assert context is not None
    assert context.stage == "position"
    assert context.input_index == 0
    assert context.input_position == pytest.approx(0.5)
    assert context.residual_norm == pytest.approx(0.5)
    assert context.condition_number == pytest.approx(1.0)
    assert context.min_singular_value == pytest.approx(1.0)
    assert context.rank == 3
    assert context.attempted_strategies == ("initial_guess",)
    assert context.corrector_attempts == 1


def test_failed_requested_sample_reports_recovery_path(monkeypatch):
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
        if input_value == pytest.approx(0.0):
            return np.array([0.0, 0.0, 0.0])
        raise KinematicSolveError(
            f"unreachable {input_value}",
            context=SolveFailureContext(
                stage="position",
                input_index=input_index,
                input_position=input_value,
                residual_norm=1.0,
            ),
        )

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_solve_configuration)
    monkeypatch.setattr(
        "kimech.solver._predict_next_configuration",
        lambda *args, **kwargs: args[4].copy(),
    )

    with pytest.raises(KinematicSolveError) as captured:
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=[0.0, 0.2],
            ),
            initial_guess={link: (0.0, 0.0, 0.0)},
        )

    context = captured.value.context
    assert context is not None
    assert context.input_index == 1
    assert context.input_position == pytest.approx(0.2)
    assert context.attempted_strategies == ("warm_start", "subdivision")
    assert context.corrector_attempts > 1
    assert "unreachable 0.2" in str(captured.value)


def test_velocity_failure_exposes_structured_context(monkeypatch):
    mechanism, link, joint = _single_revolute()

    def fail_solve(matrix, rhs):
        raise np.linalg.LinAlgError("deliberate singular matrix")

    monkeypatch.setattr("kimech._differential.np.linalg.solve", fail_solve)

    with pytest.raises(KinematicSolveError) as captured:
        solve(
            mechanism,
            driver=KinematicDriver(
                joint,
                position=0.5,
                velocity=1.0,
            ),
            initial_guess={link: (0.0, 0.0, 0.0)},
        )

    context = captured.value.context
    assert context is not None
    assert context.stage == "velocity"
    assert context.input_index == 0
    assert context.input_position == pytest.approx(0.5)
    assert context.condition_number == pytest.approx(1.0)
    assert context.min_singular_value == pytest.approx(1.0)
    assert context.rank == 3


def test_diagnostics_summary_reports_descriptive_extrema_and_effort():
    diagnostics = SolveDiagnostics(
        [2.0, 10.0, 4.0],
        [0.5, 0.1, 0.25],
        [3, 2, 3],
        subdivision_counts=[0, 2, 1],
        strategies=["initial_guess", "subdivision", "warm_start"],
        corrector_attempts=[1, 7, 2],
        residual_norms=[1e-12, 2e-10, 4e-11],
    )

    summary = diagnostics.summary()

    assert isinstance(summary, SolveDiagnosticSummary)
    assert summary.sample_count == 3
    assert summary.worst_condition_index == 1
    assert summary.worst_condition_number == pytest.approx(10.0)
    assert summary.minimum_singular_value_index == 1
    assert summary.minimum_singular_value == pytest.approx(0.1)
    assert summary.minimum_rank == 2
    assert summary.max_subdivision_index == 1
    assert summary.max_subdivision_count == 2
    assert summary.max_corrector_attempt_index == 1
    assert summary.max_corrector_attempts == 7
    assert summary.strategy_counts == (
        ("initial_guess", 1),
        ("warm_start", 1),
        ("subdivision", 1),
    )


def test_diagnostics_summary_handles_empty_and_optional_process_histories():
    empty = SolveDiagnostics([], [], [])
    summary = empty.summary()

    assert summary.sample_count == 0
    assert summary.worst_condition_index is None
    assert summary.minimum_singular_value_index is None
    assert summary.minimum_rank is None
    assert summary.strategy_counts == ()

    diagnostics = SolveDiagnostics([2.0], [0.5], [3])
    summary = diagnostics.summary()
    assert summary.max_subdivision_count is None
    assert summary.max_corrector_attempts is None
    assert summary.strategy_counts == ()
