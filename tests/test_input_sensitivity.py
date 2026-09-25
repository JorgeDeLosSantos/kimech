import numpy as np
import pytest

from kimech import (
    DriverSensitivity,
    KinematicDriver,
    KinematicSolveError,
    Mechanism,
    SolveFailureContext,
    driver_sensitivity,
    solve,
)


def _single_revolute():
    mechanism = Mechanism("single_revolute")
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("O", (0.0, 0.0))
    point = link.add_point("P", (2.0, 0.0))
    joint = mechanism.revolute(fixed, pivot, name="input")
    guess = {link: (0.0, 0.0, 0.2)}
    return mechanism, link, point, joint, guess


def _single_prismatic():
    mechanism = Mechanism("single_prismatic")
    guide = mechanism.ground.add_point("G", (0.0, 0.0))
    slider = mechanism.add_link("slider")
    slider_point = slider.add_point("G", (0.0, 0.0))
    point = slider.add_point("P", (0.5, 0.25))
    joint = mechanism.prismatic(
        guide,
        slider_point,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="input",
    )
    guess = {slider: (0.3, 0.0, 0.0)}
    return mechanism, slider, point, joint, guess


def _four_bar(scale=1.0):
    mechanism = Mechanism("four_bar")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30 * scale, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08 * scale, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22 * scale, 0.0))
    coupler_p = coupler.add_point("P", (0.10 * scale, 0.05 * scale))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18 * scale, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    mechanism.revolute(crank_b, coupler_b, name="crank_coupler")
    rocker_joint = mechanism.revolute(coupler_c, rocker_c, name="coupler_rocker")
    mechanism.revolute(rocker_d, ground_d, name="rocker_ground")

    guess = {
        crank: (0.0, 0.0, 0.9),
        coupler: (0.05 * scale, 0.06 * scale, 0.2),
        rocker: (0.30 * scale, 0.0, 2.2),
    }
    return mechanism, input_joint, rocker_joint, coupler_p, guess


def test_revolute_driver_sensitivity_is_exact_and_derives_point_motion():
    mechanism, link, point, joint, guess = _single_revolute()
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.3, 0.7],
        ),
        initial_guess=guess,
    )

    sensitivity = driver_sensitivity(solution)

    assert isinstance(sensitivity, DriverSensitivity)
    np.testing.assert_allclose(
        sensitivity.coordinate_derivatives,
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitivity.body_pose_derivatives(link),
        sensitivity.coordinate_derivatives,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitivity.joint_coordinate_derivatives(joint),
        np.ones(2),
        atol=1e-12,
    )

    expected = np.asarray(
        [
            [-2.0 * np.sin(0.3), 2.0 * np.cos(0.3)],
            [-2.0 * np.sin(0.7), 2.0 * np.cos(0.7)],
        ]
    )
    np.testing.assert_allclose(
        sensitivity.point_position_derivatives(point),
        expected,
        atol=1e-12,
    )


def test_prismatic_driver_sensitivity_is_exact():
    mechanism, slider, point, joint, guess = _single_prismatic()
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.2, 0.8],
        ),
        initial_guess=guess,
    )

    sensitivity = driver_sensitivity(solution)

    np.testing.assert_allclose(
        sensitivity.coordinate_derivatives,
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitivity.body_pose_derivatives(slider),
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitivity.point_position_derivatives(point),
        [[1.0, 0.0], [1.0, 0.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitivity.joint_coordinate_derivatives(joint),
        np.ones(2),
        atol=1e-12,
    )


def test_four_bar_sensitivity_matches_central_finite_difference():
    mechanism, input_joint, rocker_joint, point, guess = _four_bar()
    center = 0.9
    # Keep the finite-difference step comfortably above the nonlinear
    # position-solve residual floor so solve error is not amplified by 1/h.
    step = 1e-4
    values = [center - step, center, center + step]

    solution = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values,
        ),
        initial_guess=guess,
    )
    sensitivity = driver_sensitivity(solution)

    finite_difference_coordinates = (
        solution.coordinates[2] - solution.coordinates[0]
    ) / (2.0 * step)
    np.testing.assert_allclose(
        sensitivity.coordinate_derivatives[1],
        finite_difference_coordinates,
        rtol=2e-5,
        atol=2e-7,
    )

    point_positions = solution.point_positions(point)
    finite_difference_point = (
        point_positions[2] - point_positions[0]
    ) / (2.0 * step)
    np.testing.assert_allclose(
        sensitivity.point_position_derivatives(point)[1],
        finite_difference_point,
        rtol=2e-5,
        atol=2e-7,
    )

    rocker_coordinates = solution.joint_coordinates(rocker_joint)
    finite_difference_joint = (
        rocker_coordinates[2] - rocker_coordinates[0]
    ) / (2.0 * step)
    assert sensitivity.joint_coordinate_derivatives(rocker_joint)[1] == pytest.approx(
        finite_difference_joint,
        rel=2e-5,
        abs=2e-7,
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_revolute_driver_sensitivity_scales_consistently_with_geometry(scale):
    mechanism, input_joint, _, point, guess = _four_bar(scale)
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=[0.9],
        ),
        initial_guess=guess,
    )
    sensitivity = driver_sensitivity(solution)

    reference_mechanism, reference_input, _, reference_point, reference_guess = _four_bar(1.0)
    reference_solution = solve(
        reference_mechanism,
        driver=KinematicDriver(
            reference_input,
            position=[0.9],
        ),
        initial_guess=reference_guess,
    )
    reference = driver_sensitivity(reference_solution)

    np.testing.assert_allclose(
        sensitivity.point_position_derivatives(point),
        scale * reference.point_position_derivatives(reference_point),
        rtol=1e-9,
        atol=1e-10 * max(1.0, scale),
    )

    angular_columns = sensitivity.coordinate_derivatives[:, 2::3]
    reference_angular = reference.coordinate_derivatives[:, 2::3]
    np.testing.assert_allclose(
        angular_columns,
        reference_angular,
        rtol=1e-9,
        atol=1e-10,
    )


def test_returned_sensitivity_arrays_are_safe_copies():
    mechanism, _, _, joint, guess = _single_revolute()
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.5],
        ),
        initial_guess=guess,
    )
    sensitivity = driver_sensitivity(solution)

    values = sensitivity.coordinate_derivatives
    values[0, 2] = 99.0

    assert sensitivity.coordinate_derivatives[0, 2] == pytest.approx(1.0)


def test_sensitivity_uses_solution_joint_snapshot_after_mechanism_extension():
    mechanism, link, _, joint, guess = _single_revolute()
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.5],
        ),
        initial_guess=guess,
    )

    extension = mechanism.add_link("extension")
    ground_e = mechanism.ground.add_point("E", (1.0, 0.0))
    extension_e = extension.add_point("E", (0.0, 0.0))
    new_joint = mechanism.revolute(ground_e, extension_e, name="new_joint")

    sensitivity = driver_sensitivity(solution)

    np.testing.assert_allclose(
        sensitivity.coordinate_derivatives,
        [[0.0, 0.0, 1.0]],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="snapshot"):
        sensitivity.joint_coordinate_derivatives(new_joint)

    assert sensitivity.body_pose_derivatives(link)[0, 2] == pytest.approx(1.0)


def test_sensitivity_failure_does_not_mutate_or_invalidate_solution(monkeypatch):
    mechanism, link, _, joint, guess = _single_revolute()
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            joint,
            position=[0.5],
        ),
        initial_guess=guess,
    )
    baseline = solution.coordinates.copy()

    def fail_tangent(*args, **kwargs):
        raise KinematicSolveError(
            "deliberate tangent failure",
            context=SolveFailureContext(
                stage="driver tangent",
                driver_index=0,
                driver_position=0.5,
            ),
        )

    monkeypatch.setattr("kimech.sensitivity.solve_driver_tangent", fail_tangent)

    with pytest.raises(KinematicSolveError, match="deliberate tangent failure"):
        driver_sensitivity(solution)

    np.testing.assert_array_equal(solution.coordinates, baseline)
    assert solution[0].body_pose(link)[2] == pytest.approx(0.5)


def test_driver_sensitivity_rejects_non_solution():
    with pytest.raises(TypeError, match="KinematicSolution"):
        driver_sensitivity(object())
