"""Analyze four-bar position, velocity, and acceleration with Kimech."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve


def build_mechanism():
    """Build a four-bar linkage and a suitable initial pose estimate."""
    mechanism = Mechanism("four_bar_analysis")
    ground = mechanism.ground

    ground_a = ground.add_point("A", (0.0, 0.0))
    ground_d = ground.add_point("D", (0.30, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22, 0.0))
    point_p = coupler.add_point("P", (0.10, 0.05))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)

    initial_guess = {
        crank: (0.0, 0.0, 0.8),
        coupler: (0.05, 0.06, 0.2),
        rocker: (0.30, 0.0, 2.2),
    }
    return mechanism, input_joint, rocker, point_p, initial_guess


def main():
    mechanism, input_joint, rocker, point_p, initial_guess = build_mechanism()

    # The prescribed crank rotates at constant angular velocity.
    input_angle = np.linspace(0.8, 0.8 + 2 * np.pi, 181)
    input_angular_velocity = 1.0
    input_angular_acceleration = 0.0

    solution = solve(
        mechanism,
        input=input_joint,
        values=input_angle,
        input_velocity=input_angular_velocity,
        input_acceleration=input_angular_acceleration,
        initial_guess=initial_guess,
    )

    rocker_pose = solution.body_poses(rocker)
    rocker_velocity = solution.body_velocities(rocker)
    rocker_acceleration = solution.body_accelerations(rocker)

    rocker_angle = rocker_pose[:, 2]
    rocker_angular_velocity = rocker_velocity[:, 2]
    rocker_angular_acceleration = rocker_acceleration[:, 2]

    point_velocity = solution.point_velocities(point_p)
    point_acceleration = solution.point_accelerations(point_p)
    point_speed = np.linalg.norm(point_velocity, axis=1)
    point_acceleration_magnitude = np.linalg.norm(point_acceleration, axis=1)

    print("Four-bar differential analysis")
    print(f"Solved {len(solution)} configurations")
    print(
        "Rocker angular velocity range: "
        f"{rocker_angular_velocity.min():.6f} -> "
        f"{rocker_angular_velocity.max():.6f} rad/s"
    )
    print(
        "Rocker angular acceleration range: "
        f"{rocker_angular_acceleration.min():.6f} -> "
        f"{rocker_angular_acceleration.max():.6f} rad/s^2"
    )
    print(f"Maximum speed of P: {point_speed.max():.6f}")
    print(f"Maximum acceleration magnitude of P: {point_acceleration_magnitude.max():.6f}")

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(input_angle, rocker_angle)
    axes[0].set_ylabel("angle [rad]")
    axes[0].set_title("Rocker kinematics")

    axes[1].plot(input_angle, rocker_angular_velocity)
    axes[1].set_ylabel("angular velocity [rad/s]")

    axes[2].plot(input_angle, rocker_angular_acceleration)
    axes[2].set_xlabel("input crank angle [rad]")
    axes[2].set_ylabel("angular acceleration [rad/s^2]")

    fig.tight_layout()

    fig_point, axes_point = plt.subplots(2, 1, sharex=True)
    axes_point[0].plot(input_angle, point_speed)
    axes_point[0].set_ylabel("speed")
    axes_point[0].set_title("Coupler point P")

    axes_point[1].plot(input_angle, point_acceleration_magnitude)
    axes_point[1].set_xlabel("input crank angle [rad]")
    axes_point[1].set_ylabel("acceleration magnitude")

    fig_point.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
