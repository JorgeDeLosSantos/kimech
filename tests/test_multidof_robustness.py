"""Cross-scale and non-monotone Multi-DOF acceptance with analytic references."""

import numpy as np
import pytest

from kimech import KinematicDriver, Mechanism, solve


def _unit(theta):
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _normal(theta):
    return np.column_stack((-np.sin(theta), np.cos(theta)))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_serial_2r_mixed_direction_input_paths_are_scale_invariant(scale):
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("base", (0.0, 0.0))
    first = mechanism.add_link("first")
    first_base = first.add_point("base", (0.0, 0.0))
    first_end = first.add_point("elbow", (2 * scale, 0.0))
    second = mechanism.add_link("second")
    second_base = second.add_point("elbow", (0.0, 0.0))
    tip = second.add_point("tip", (scale, 0.0))
    j1 = mechanism.revolute(fixed, first_base)
    j2 = mechanism.revolute(first_end, second_base)

    # Both coordinates change in non-monotone, occasionally opposite directions.
    a = np.array([0.2, 0.57, 0.1, -0.16, 0.39, 0.24])
    b = np.array([0.32, -0.25, 0.49, 0.12, -0.35, 0.26])
    wa = np.array([0.8, -0.3, 0.2, -1.1, 0.3, 0.5])
    wb = np.array([-0.2, 0.9, -0.4, 0.3, -0.6, 0.8])
    aa = np.array([0.1, -0.2, 0.4, 0.3, -0.5, 0.2])
    ab = np.array([-0.3, 0.2, 0.1, -0.4, 0.6, -0.1])
    result = solve(
        mechanism,
        drivers=[
            KinematicDriver(j1, position=a, velocity=wa, acceleration=aa),
            KinematicDriver(j2, position=b, velocity=wb, acceleration=ab),
        ],
        initial_guess={
            first: (0., 0., a[0]),
            second: (2*scale*np.cos(a[0]), 2*scale*np.sin(a[0]), a[0]+b[0]),
        },
        time=np.linspace(0., 1., len(a)),
    )
    expected_p = scale * (2 * _unit(a) + _unit(a+b))
    expected_v = scale * (2*wa[:, None]*_normal(a) + (wa+wb)[:, None]*_normal(a+b))
    expected_acc = scale * (
        2*aa[:, None]*_normal(a) - 2*wa[:, None]**2*_unit(a)
        + (aa+ab)[:, None]*_normal(a+b)
        - (wa+wb)[:, None]**2*_unit(a+b)
    )
    np.testing.assert_allclose(result.point_positions(tip), expected_p, rtol=2e-8, atol=2e-9*scale)
    np.testing.assert_allclose(result.point_velocities(tip), expected_v, rtol=2e-8, atol=2e-9*scale)
    np.testing.assert_allclose(result.point_accelerations(tip), expected_acc, rtol=2e-8, atol=2e-9*scale)
    np.testing.assert_allclose(result.joint_coordinates(j1), a, atol=2e-8)
    np.testing.assert_allclose(result.joint_coordinates(j2), b, atol=2e-8)
    assert np.all(result.diagnostics.rank_issues == "regular")
    assert np.max(result.diagnostics.residual_norms) < 1e-8
    assert len(result) == len(a)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_mixed_revolute_prismatic_differentials_scale_consistently(scale):
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("origin", (0., 0.))
    arm = mechanism.add_link("arm")
    pivot = arm.add_point("base", (0., 0.))
    guide = arm.add_point("guide", (1.5*scale, 0.))
    slider = mechanism.add_link("slider")
    carriage = slider.add_point("carriage", (0., 0.))
    r = mechanism.revolute(fixed, pivot)
    p = mechanism.prismatic(guide, carriage, axis_a=(1., 0.), axis_b=(1., 0.))

    theta = np.array([0.2, 0.59, 0.08, -0.14, 0.34])
    travel = np.array([0.4, 0.7, -0.1, 0.3, 0.1])
    omega = np.array([1.1, -0.6, 0.2, 0.8, -0.3])
    speed = np.array([0.5, -0.7, 0.8, -0.2, 0.1])
    alpha = np.array([-0.2, 0.7, 0.4, -0.5, 0.6])
    acceleration = np.array([0.8, -0.1, 0.3, -0.4, -0.2])
    length = 1.5 + travel
    result = solve(
        mechanism,
        drivers=[
            KinematicDriver(r, position=theta, velocity=omega, acceleration=alpha),
            KinematicDriver(p, position=scale*travel, velocity=scale*speed,
                            acceleration=scale*acceleration),
        ],
        initial_guess={
            arm: (0., 0., theta[0]),
            slider: (*((scale*length[0])*_unit(theta[0:1])[0]), theta[0]),
        },
    )
    expected_p = scale*length[:, None]*_unit(theta)
    expected_v = scale*(
        speed[:, None]*_unit(theta)
        + (length*omega)[:, None]*_normal(theta)
    )
    expected_a = scale*(
        (acceleration-length*omega**2)[:, None]*_unit(theta)
        + (2*speed*omega + length*alpha)[:, None]*_normal(theta)
    )
    np.testing.assert_allclose(result.body_poses(slider)[:, :2], expected_p, rtol=1e-8, atol=2e-9*scale)
    np.testing.assert_allclose(result.body_velocities(slider)[:, :2], expected_v, rtol=1e-8, atol=2e-9*scale)
    np.testing.assert_allclose(result.body_accelerations(slider)[:, :2], expected_a, rtol=1e-8, atol=2e-9*scale)
    assert np.all(result.diagnostics.rank_issues == "regular")
    np.testing.assert_allclose(result.joint_coordinates(p), scale*travel, rtol=1e-8, atol=2e-9*scale)


def test_five_bar_approach_to_toggle_reports_worsening_condition_without_branch_jump():
    """Track a physically valid closed chain toward (but not into) a toggle."""
    mechanism = Mechanism()
    lg = mechanism.ground.add_point("left", (-1., 0.))
    rg = mechanism.ground.add_point("right", (1., 0.))
    lcrank = mechanism.add_link("left_crank")
    rcrank = mechanism.add_link("right_crank")
    lc = mechanism.add_link("left_coupler")
    rc = mechanism.add_link("right_coupler")
    la = lcrank.add_point("pivot", (0., 0.))
    lb = lcrank.add_point("tip", (1.5, 0.))
    ra = rcrank.add_point("pivot", (0., 0.))
    rb = rcrank.add_point("tip", (1.5, 0.))
    lcbase = lc.add_point("start", (0., 0.))
    lctip = lc.add_point("apex", (1.5, 0.))
    rcbase = rc.add_point("start", (0., 0.))
    rctip = rc.add_point("apex", (1.5, 0.))
    jleft = mechanism.revolute(lg, la)
    mechanism.revolute(lb, lcbase)
    mechanism.revolute(lctip, rctip)
    mechanism.revolute(rb, rcbase)
    jright = mechanism.revolute(rg, ra)

    critical = float(np.arcsin(np.sqrt(5.) / 3.))
    angles = np.array([0.65, 0.77, 0.81, critical - 0.005])
    first = float(angles[0])
    pleft = np.array([-1. + 1.5*np.cos(first), 1.5*np.sin(first)])
    pright = np.array([1. + 1.5*np.cos(first), -1.5*np.sin(first)])
    delta = pright - pleft
    distance = np.linalg.norm(delta)
    height = np.sqrt(1.5**2 - (distance/2)**2)
    apex = 0.5*(pleft+pright) + height*np.array([-delta[1],delta[0]])/distance
    angle_lc = np.arctan2(apex[1]-pleft[1], apex[0]-pleft[0])
    angle_rc = np.arctan2(apex[1]-pright[1], apex[0]-pright[0])
    solution = solve(
        mechanism,
        drivers=[
            KinematicDriver(jleft, position=angles),
            KinematicDriver(jright, position=-angles),
        ],
        initial_guess={
            lcrank: (-1., 0., first),
            rcrank: (1., 0., -first),
            lc: (*pleft, angle_lc),
            rc: (*pright, angle_rc),
        },
    )
    assert len(solution) == len(angles)
    np.testing.assert_allclose(solution.point_positions(lctip),
                               solution.point_positions(rctip), atol=2e-8)
    assert np.all(solution.diagnostics.rank_issues == "regular")
    assert solution.diagnostics.condition_numbers[-1] > solution.diagnostics.condition_numbers[0]
    assert solution.diagnostics.min_singular_values[-1] < solution.diagnostics.min_singular_values[0]
