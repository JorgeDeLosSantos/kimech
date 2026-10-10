"""Analytic Multi-DOF position, velocity and acceleration example: planar 2R."""

import numpy as np

from kimech import KinematicDriver, Mechanism, solve


def main():
    mechanism = Mechanism("serial_2r")

    ground = mechanism.ground.add_point("base", (0.0, 0.0))
    link1 = mechanism.add_link("first")
    link1_base = link1.add_point("base", (0.0, 0.0))
    link1_end = link1.add_point("elbow", (2.0, 0.0))

    link2 = mechanism.add_link("second")
    link2_base = link2.add_point("elbow", (0.0, 0.0))
    tip = link2.add_point("tip", (1.0, 0.0))

    joint1 = mechanism.revolute(ground, link1_base)
    joint2 = mechanism.revolute(link1_end, link2_base)

    time = np.linspace(0.0, 1.0, 21)
    theta1 = 0.2 + 0.4 * time + 0.1 * time**2
    theta2 = -0.1 + 0.3 * time - 0.15 * time**2
    omega1 = 0.4 + 0.2 * time
    omega2 = 0.3 - 0.3 * time
    alpha1 = 0.2
    alpha2 = -0.3

    first = KinematicDriver(
        joint1, position=theta1, velocity=omega1, acceleration=alpha1
    )
    second = KinematicDriver(
        joint2, position=theta2, velocity=omega2, acceleration=alpha2
    )

    initial_guess = {
        link1: (0.0, 0.0, theta1[0]),
        link2: (
            2.0 * np.cos(theta1[0]),
            2.0 * np.sin(theta1[0]),
            theta1[0] + theta2[0],
        ),
    }
    solution = solve(
        mechanism,
        drivers=[first, second],
        initial_guess=initial_guess,
        time=time,
    )

    combined = theta1 + theta2
    combined_omega = omega1 + omega2
    unit = lambda angle: np.column_stack((np.cos(angle), np.sin(angle)))
    normal = lambda angle: np.column_stack((-np.sin(angle), np.cos(angle)))

    expected_position = 2.0 * unit(theta1) + unit(combined)
    expected_velocity = (
        2.0 * omega1[:, None] * normal(theta1)
        + combined_omega[:, None] * normal(combined)
    )
    expected_acceleration = (
        2.0 * alpha1 * normal(theta1)
        - 2.0 * omega1[:, None]**2 * unit(theta1)
        + (alpha1 + alpha2) * normal(combined)
        - combined_omega[:, None]**2 * unit(combined)
    )

    np.testing.assert_allclose(solution.point_positions(tip), expected_position, atol=1e-9)
    np.testing.assert_allclose(solution.point_velocities(tip), expected_velocity, atol=1e-9)
    np.testing.assert_allclose(solution.point_accelerations(tip), expected_acceleration, atol=1e-9)

    print(f"Samples: {len(solution)} | drivers: {len(solution.drivers)}")
    print("Final tip position:", solution.point_positions(tip)[-1])
    print("Final tip velocity:", solution.point_velocities(tip)[-1])
    print("Final tip acceleration:", solution.point_accelerations(tip)[-1])


if __name__ == "__main__":
    main()
