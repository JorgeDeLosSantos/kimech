import numpy as np

from kimech import KinematicDriver, Mechanism, solve


def _four_bar():
    mechanism = Mechanism("four_bar_time_acceptance")

    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22, 0.0))
    coupler_p = coupler.add_point("P", (0.10, 0.05))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    crank_joint = mechanism.revolute(ground_a, crank_a, name="crank_driver")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)

    initial_guess = {
        crank: (0.0, 0.0, np.deg2rad(30.0)),
        coupler: (0.05, 0.06, 0.2),
        rocker: (0.30, 0.0, 2.2),
    }
    return mechanism, crank_joint, coupler_p, initial_guess


def test_constant_speed_four_bar_time_history_matches_equivalent_kinematic_sweep():
    mechanism, crank_joint, coupler_p, initial_guess = _four_bar()

    time = np.linspace(0.0, 2.0, 201)
    theta0 = np.deg2rad(30.0)
    omega = 5.0
    alpha = 0.0
    theta = theta0 + omega * time

    driver = KinematicDriver(
        crank_joint,
        position=theta,
        velocity=omega,
        acceleration=alpha,
    )

    timed = solve(
        mechanism,
        driver=driver,
        time=time,
        initial_guess=initial_guess,
    )
    untimed = solve(
        mechanism,
        driver=driver,
        initial_guess=initial_guess,
    )

    np.testing.assert_array_equal(timed.time, time)
    np.testing.assert_array_equal(timed.driver.position, theta)
    np.testing.assert_array_equal(timed.driver.velocity, np.full_like(time, omega))
    np.testing.assert_array_equal(timed.driver.acceleration, np.full_like(time, alpha))

    np.testing.assert_allclose(
        timed.joint_coordinates(crank_joint),
        theta,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        timed.joint_velocities(crank_joint),
        omega,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        timed.joint_accelerations(crank_joint),
        alpha,
        atol=1e-10,
    )

    np.testing.assert_array_equal(timed.coordinates, untimed.coordinates)
    np.testing.assert_array_equal(
        timed.coordinate_velocities,
        untimed.coordinate_velocities,
    )
    np.testing.assert_array_equal(
        timed.coordinate_accelerations,
        untimed.coordinate_accelerations,
    )
    np.testing.assert_array_equal(
        timed.point_positions(coupler_p),
        untimed.point_positions(coupler_p),
    )

    assert timed[0].time == 0.0
    assert timed[-1].time == 2.0
    assert timed[-1].driver.position == theta[-1]
