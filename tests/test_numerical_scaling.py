from types import SimpleNamespace

import numpy as np
import pytest

from kimech import KinematicDriver, Mechanism, solve
from kimech._scaling import build_numerical_scaling


def _problem_four_bar(scale: float):
    mechanism = Mechanism("problem_four_bar")
    ground_o = mechanism.ground.add_point("O", (0.0, 0.0))
    ground_c = mechanism.ground.add_point("C", (120.0 * scale, 0.0))

    link_2 = mechanism.add_link("link_2")
    link_2_o = link_2.add_point("O", (-25.0 * scale, 0.0))
    link_2_a = link_2.add_point("A", (25.0 * scale, 0.0))

    link_3 = mechanism.add_link("link_3")
    link_3_a = link_3.add_point("A", (0.0, 0.0))
    link_3_b = link_3.add_point("B", (200.0 * scale, 0.0))

    link_4 = mechanism.add_link("link_4")
    link_4_b = link_4.add_point("B", (200.0 * scale, 0.0))
    link_4_c = link_4.add_point("C", (0.0, 0.0))

    input_joint = mechanism.revolute(ground_o, link_2_o, name="input")
    mechanism.revolute(link_2_a, link_3_a)
    mechanism.revolute(link_3_b, link_4_b)
    mechanism.revolute(link_4_c, ground_c)

    guess = {
        link_2: (25.0 * scale, 0.0, 0.0),
        link_3: (50.0 * scale, 0.0, 1.4),
        link_4: (120.0 * scale, 0.0, 1.75),
    }
    return mechanism, input_joint, guess


def _pure_prismatic(scale: float):
    mechanism = Mechanism("pure_prismatic")
    fixed = mechanism.ground.add_point("G", (0.0, 0.0))
    slider = mechanism.add_link("slider")
    moving = slider.add_point("G", (0.0, 0.0))
    input_joint = mechanism.prismatic(
        fixed,
        moving,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="input",
    )
    values = scale * np.linspace(0.0, 100.0, 101)
    guess = {slider: (0.0, 0.0, 0.0)}
    return mechanism, slider, input_joint, values, guess


def _baseline_four_bar(scale: float):
    mechanism = Mechanism("baseline_four_bar")
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


@pytest.mark.parametrize("scale", [1e-3, 1e-1, 1.0, 1e3])
def test_problem_four_bar_sweep_is_scale_invariant(scale):
    mechanism, input_joint, guess = _problem_four_bar(scale)
    values = np.linspace(0.0, 2.0 * np.pi, 361)

    solution = solve(
        mechanism,
        input_joint=input_joint,
        input_position=values,
        initial_guess=guess,
    )

    assert len(solution) == len(values)
    np.testing.assert_allclose(solution.input_positions, values)


def test_characteristic_length_scales_with_problem_geometry():
    base_mechanism, base_input, _ = _problem_four_bar(1.0)
    scaled_mechanism, scaled_input, _ = _problem_four_bar(1e3)
    values = np.linspace(0.0, 2.0 * np.pi, 361)

    base = build_numerical_scaling(
        base_mechanism,
        base_mechanism.links,
        base_mechanism.joints,
        KinematicDriver(base_input, position=values),
    )
    scaled = build_numerical_scaling(
        scaled_mechanism,
        scaled_mechanism.links,
        scaled_mechanism.joints,
        KinematicDriver(scaled_input, position=values),
    )

    assert base.characteristic_length == pytest.approx(200.0)
    assert scaled.characteristic_length == pytest.approx(200000.0)
    assert scaled.characteristic_length / base.characteristic_length == pytest.approx(1e3)


def test_prismatic_sweep_accepts_verified_root_even_if_hybr_reports_no_progress(monkeypatch):
    mechanism, slider, input_joint, values, guess = _pure_prismatic(1.0)
    original_root = __import__("scipy").optimize.root

    def false_negative_root(fun, x0, *, jac, method):
        result = original_root(fun, x0, jac=jac, method=method)
        return SimpleNamespace(
            success=False,
            message="deliberate false-negative status",
            x=result.x,
        )

    monkeypatch.setattr("kimech.solver.optimize.root", false_negative_root)

    solution = solve(
        mechanism,
        input_joint=input_joint,
        input_position=values,
        initial_guess=guess,
    )

    np.testing.assert_allclose(solution.body_poses(slider)[:, 0], values, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_differential_solution_preserves_physical_scaling(scale):
    mechanism, input_joint, guess = _baseline_four_bar(scale)
    theta = 1.0
    omega = 1.3
    alpha = -0.4

    config = solve(
        mechanism,
        input_joint=input_joint,
        input_position=theta,
        input_velocity=omega,
        input_acceleration=alpha,
        initial_guess=guess,
    )[0]

    assert np.all(np.isfinite(config.coordinates))
    assert np.all(np.isfinite(config.coordinate_velocities))
    assert np.all(np.isfinite(config.coordinate_accelerations))
    assert config.joint_velocity(input_joint) == pytest.approx(omega)
    assert config.joint_acceleration(input_joint) == pytest.approx(alpha)
