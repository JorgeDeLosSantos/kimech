import numpy as np

from kimech import Mechanism, KinematicDriver, solve


def _wrapped_angle_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _four_bar():
    mechanism = Mechanism("four_bar_0_4_acceptance")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22, 0.0))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="crank_input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)

    initial_guess = {
        crank: (0.0, 0.0, 0.8),
        coupler: (0.05, 0.06, 0.2),
        rocker: (0.30, 0.0, 2.2),
    }
    return mechanism, input_joint, initial_guess


def _slider_crank():
    mechanism = Mechanism("slider_crank_0_4_acceptance")
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
    slider_joint = mechanism.prismatic(
        guide,
        slider_guide,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="slider_input",
    )

    initial_guess = {
        crank: (0.0, 0.0, 0.7),
        rod: (0.06, 0.05, -0.2),
        slider: (0.30, 0.0, 0.0),
    }
    return mechanism, crank_joint, slider_joint, initial_guess


def test_four_bar_full_cycle_is_direction_consistent_with_valid_process_diagnostics():
    mechanism, input_joint, initial_guess = _four_bar()
    values = np.linspace(0.8, 0.8 + 2.0 * np.pi, 73)

    forward = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values,
        ),
        initial_guess=initial_guess,
    )
    reverse = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values[::-1],
        ),
        initial_guess=forward[-1],
    )[::-1]

    forward_diagnostics = forward.diagnostics
    reverse_diagnostics = reverse.diagnostics
    assert forward_diagnostics is not None
    assert reverse_diagnostics is not None

    np.testing.assert_array_equal(forward.driver.position, reverse.driver.position)
    assert np.all(forward_diagnostics.ranks == 9)
    assert np.all(reverse_diagnostics.ranks == 9)
    assert np.all(forward_diagnostics.residual_norms <= 1e-9)
    assert np.all(reverse_diagnostics.residual_norms <= 1e-9)
    assert np.all(forward_diagnostics.corrector_attempts >= 1)
    assert np.all(reverse_diagnostics.corrector_attempts >= 1)

    allowed = {"initial_guess", "predictor", "warm_start", "subdivision"}
    assert set(forward_diagnostics.strategies).issubset(allowed)
    assert set(reverse_diagnostics.strategies).issubset(allowed)

    for link in mechanism.links:
        forward_poses = forward.body_poses(link)
        reverse_poses = reverse.body_poses(link)

        np.testing.assert_allclose(
            forward_poses[:, :2],
            reverse_poses[:, :2],
            atol=1e-9,
        )
        angular_error = np.abs(
            _wrapped_angle_difference(
                forward_poses[:, 2],
                reverse_poses[:, 2],
            )
        )
        assert np.max(angular_error) < 1e-9


def test_slider_crank_dead_center_diagnostics_depend_on_selected_driver():
    mechanism, crank_joint, slider_joint, initial_guess = _slider_crank()

    approach = solve(
        mechanism,
        driver=KinematicDriver(
            crank_joint,
            position=np.linspace(0.7, 0.0, 40),
        ),
        initial_guess=initial_guess,
    )
    dead_center = approach[-1]

    crank_driven = solve(
        mechanism,
        driver=KinematicDriver(
            crank_joint,
            position=0.0,
        ),
        initial_guess=dead_center,
    )
    slider_position = dead_center.joint_coordinate(slider_joint)
    slider_driven = solve(
        mechanism,
        driver=KinematicDriver(
            slider_joint,
            position=slider_position,
        ),
        initial_guess=dead_center,
    )

    crank_diag = crank_driven.diagnostics
    slider_diag = slider_driven.diagnostics
    assert crank_diag is not None
    assert slider_diag is not None

    assert crank_diag.ranks[0] == 9
    assert crank_diag.condition_numbers[0] < 10.0
    assert crank_diag.min_singular_values[0] > 0.1

    assert slider_diag.ranks[0] < crank_diag.ranks[0]
    assert slider_diag.condition_numbers[0] > 1e12
    assert slider_diag.min_singular_values[0] < 1e-12

    assert crank_diag.residual_norms[0] <= 1e-9
    assert slider_diag.residual_norms[0] <= 1e-9
