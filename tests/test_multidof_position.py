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
