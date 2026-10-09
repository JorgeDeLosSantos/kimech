import numpy as np
import pytest

from kimech import KinematicDriver, Mechanism, solve, InvalidModelError


def _serial_2r():
    mechanism = Mechanism()
    base = mechanism.ground.add_point("base", (0.0, 0.0))
    first = mechanism.add_link("first")
    first_joint = first.add_point("A", (0.0, 0.0))
    first_tip = first.add_point("B", (2.0, 0.0))
    second = mechanism.add_link("second")
    second_joint = second.add_point("A", (0.0, 0.0))
    second_tip = second.add_point("C", (1.0, 0.0))
    j1 = mechanism.revolute(base, first_joint)
    j2 = mechanism.revolute(first_tip, second_joint)
    return mechanism, first, second, second_tip, j1, j2


def test_serial_2r_two_drivers_position_history():
    mechanism, first, second, tip, j1, j2 = _serial_2r()
    a = np.array([0.2, 0.4, 0.6])
    b = np.array([0.3, -0.2, 0.5])
    drivers = (
        KinematicDriver(j1, position=a),
        KinematicDriver(j2, position=b),
    )
    guess = {first: (0.0, 0.0, a[0]), second: (
        2 * np.cos(a[0]), 2 * np.sin(a[0]), a[0] + b[0])}
    result = solve(mechanism, drivers=drivers, initial_guess=guess)
    assert len(result) == len(a)
    assert len(result.drivers) == 2
    np.testing.assert_allclose(result.joint_coordinates(j1), a, atol=1e-9)
    np.testing.assert_allclose(result.joint_coordinates(j2), b, atol=1e-9)
    expected = np.column_stack((
        2 * np.cos(a) + np.cos(a + b),
        2 * np.sin(a) + np.sin(a + b),
    ))
    np.testing.assert_allclose(result.point_positions(tip), expected, atol=1e-9)
    assert result.diagnostics is not None
    assert all(result.diagnostics.ranks == 6)


def test_serial_2r_single_configuration_and_time():
    mechanism, first, second, _, j1, j2 = _serial_2r()
    result = solve(
        mechanism,
        drivers=[KinematicDriver(j1, position=0.2), KinematicDriver(j2, position=0.3)],
        initial_guess={first: (0, 0, 0.2), second: (2, 0, 0.5)},
        time=1.0,
    )
    assert len(result) == 1
    assert result[0].time == pytest.approx(1.0)
    assert len(result[0].drivers) == 2


def test_serial_2r_requires_two_drivers():
    mechanism, first, second, _, j1, _ = _serial_2r()
    with pytest.raises(InvalidModelError, match="mobility 1"):
        solve(mechanism, drivers=KinematicDriver(j1, position=0.2),
              initial_guess={first:(0,0,0),second:(2,0,0)})


def test_five_bar_closed_chain_with_two_prescribed_cranks():
    mechanism = Mechanism()
    left_ground = mechanism.ground.add_point("LG", (-1.0, 0.0))
    right_ground = mechanism.ground.add_point("RG", (1.0, 0.0))
    left_crank = mechanism.add_link("left_crank")
    right_crank = mechanism.add_link("right_crank")
    left_coupler = mechanism.add_link("left_coupler")
    right_coupler = mechanism.add_link("right_coupler")

    la = left_crank.add_point("A", (0.0, 0.0))
    lb = left_crank.add_point("B", (1.5, 0.0))
    ra = right_crank.add_point("A", (0.0, 0.0))
    rb = right_crank.add_point("B", (1.5, 0.0))
    lc0 = left_coupler.add_point("base", (0.0, 0.0))
    lc1 = left_coupler.add_point("tip", (1.5, 0.0))
    rc0 = right_coupler.add_point("base", (0.0, 0.0))
    rc1 = right_coupler.add_point("tip", (1.5, 0.0))
    jleft = mechanism.revolute(left_ground, la)
    mechanism.revolute(lb, lc0)
    mechanism.revolute(lc1, rc1)
    mechanism.revolute(rb, rc0)
    jright = mechanism.revolute(right_ground, ra)

    theta_left = 0.6
    theta_right = np.pi - 0.6
    pleft = np.array([-1.0 + 1.5 * np.cos(theta_left), 1.5 * np.sin(theta_left)])
    pright = np.array([1.0 + 1.5 * np.cos(theta_right), 1.5 * np.sin(theta_right)])
    middle = (pleft + pright) / 2
    half_distance = np.linalg.norm(pright - pleft) / 2
    apex = middle + np.array([0.0, np.sqrt(1.5 ** 2 - half_distance ** 2)])
    aleft = np.arctan2(*(apex - pleft)[::-1])
    aright = np.arctan2(*(apex - pright)[::-1])

    result = solve(
        mechanism,
        drivers=[
            KinematicDriver(
                jleft, position=[theta_left, theta_left + 0.02],
                velocity=[0.7, 0.8], acceleration=[0.2, -0.3],
            ),
            KinematicDriver(
                jright, position=[theta_right, theta_right - 0.02],
                velocity=[-0.4, -0.5], acceleration=[-0.1, 0.4],
            ),
        ],
        initial_guess={
            left_crank: (-1.0, 0.0, theta_left),
            right_crank: (1.0, 0.0, theta_right),
            left_coupler: (*pleft, aleft),
            right_coupler: (*pright, aright),
        },
    )
    assert len(result) == 2
    np.testing.assert_allclose(result.joint_coordinates(jleft),
                               [theta_left, theta_left + 0.02], atol=1e-8)
    np.testing.assert_allclose(result.joint_coordinates(jright),
                               [theta_right, theta_right - 0.02], atol=1e-8)
    np.testing.assert_allclose(result.point_positions(lc1),
                               result.point_positions(rc1), atol=1e-8)
    assert result.has_velocity and result.has_acceleration
    np.testing.assert_allclose(result.joint_velocities(jleft), [0.7, 0.8], atol=1e-8)
    np.testing.assert_allclose(result.joint_velocities(jright), [-0.4, -0.5], atol=1e-8)
    np.testing.assert_allclose(result.joint_accelerations(jleft), [0.2, -0.3], atol=1e-8)
    np.testing.assert_allclose(result.joint_accelerations(jright), [-0.1, 0.4], atol=1e-8)
    np.testing.assert_allclose(result.point_velocities(lc1),
                               result.point_velocities(rc1), atol=1e-8)
    np.testing.assert_allclose(result.point_accelerations(lc1),
                               result.point_accelerations(rc1), atol=1e-8)


def test_multidof_adaptive_subdivision_is_along_the_input_segment(monkeypatch):
    """A requested vector step is subdivided without exposing internal samples."""
    from kimech import KinematicSolveError

    mechanism, first, second, _, j1, j2 = _serial_2r()
    calls = []

    def fake_correct(mechanism_arg, links, joints, input_drivers, values,
                     initial_q, scaling, *, driver_index=None):
        values = np.asarray(values, dtype=float)
        calls.append(values.copy())
        # Simulate a corrector that can accept only short steps.
        if np.linalg.norm(values - [initial_q[2], initial_q[5] - initial_q[2]],
                          ord=np.inf) > 0.07 + 1e-12:
            raise KinematicSolveError("step too large")
        a, b = values
        return np.array([0.0, 0.0, a, 2.0*np.cos(a), 2.0*np.sin(a), a+b])

    monkeypatch.setattr("kimech.solver._solve_configuration", fake_correct)
    monkeypatch.setattr(
        "kimech.solver._predict_next_configuration",
        lambda mechanism, links, joints, drivers, q, start, target, scaling,
               driver_index=None: q.copy(),
    )
    result = solve(
        mechanism,
        drivers=[
            KinematicDriver(j1, position=[0.0, 0.2]),
            KinematicDriver(j2, position=[0.0, 0.1]),
        ],
        initial_guess={first: (0.0, 0.0, 0.0), second: (2.0, 0.0, 0.0)},
    )
    assert len(result) == 2
    np.testing.assert_allclose(result.joint_coordinates(j1), [0.0, 0.2])
    np.testing.assert_allclose(result.joint_coordinates(j2), [0.0, 0.1])
    assert result.diagnostics.strategies[1] == "subdivision"
    assert result.diagnostics.subdivision_counts[1] == 3
    assert result.diagnostics.corrector_attempts[1] == 7
    assert any(np.allclose(sample, [0.1, 0.05]) for sample in calls)
    assert any(np.allclose(sample, [0.05, 0.025]) for sample in calls)


def test_single_driver_and_singleton_sequence_share_solver_and_diagnostics():
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("origin", (0.0, 0.0))
    link = mechanism.add_link("link")
    pivot = link.add_point("pivot", (0.0, 0.0))
    revolute = mechanism.revolute(fixed, pivot)
    input_driver = KinematicDriver(
        revolute, position=[0.2, 0.25], velocity=1.5, acceleration=-0.4
    )
    guess = {link: (0.0, 0.0, 0.2)}
    via_single = solve(mechanism, drivers=input_driver, initial_guess=guess)
    via_plural = solve(mechanism, drivers=[input_driver], initial_guess=guess)
    np.testing.assert_allclose(via_single.coordinates, via_plural.coordinates)
    np.testing.assert_allclose(
        via_single.coordinate_velocities, via_plural.coordinate_velocities
    )
    np.testing.assert_allclose(
        via_single.coordinate_accelerations, via_plural.coordinate_accelerations
    )
    np.testing.assert_array_equal(
        via_single.diagnostics.strategies, via_plural.diagnostics.strategies
    )


def test_solve_rejects_removed_singular_driver_keyword():
    mechanism, first, second, _, j1, j2 = _serial_2r()
    guess = {first: (0., 0., 0.2), second: (2., 0., 0.5)}
    with pytest.raises(TypeError, match="driver"):
        solve(mechanism, driver=KinematicDriver(j1, position=0.2), initial_guess=guess)
