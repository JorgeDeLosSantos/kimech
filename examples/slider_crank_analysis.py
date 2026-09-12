"""Analyze slider-crank position, velocity, and acceleration with Kimech."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve


def build_mechanism():
    """Build a slider-crank and a suitable initial pose estimate."""
    mechanism = Mechanism("slider_crank_analysis")
    ground = mechanism.ground

    origin = ground.add_point("O", (0.0, 0.0))
    guide = ground.add_point("G", (0.0, 0.0))

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    connecting_rod = mechanism.add_link("connecting_rod")
    rod_b = connecting_rod.add_point("B", (0.0, 0.0))
    rod_c = connecting_rod.add_point("C", (0.24, 0.0))

    slider = mechanism.add_link("slider")
    slider_c = slider.add_point("C", (0.0, 0.0))
    slider_guide = slider.add_point("G", (0.0, 0.0))

    crank_joint = mechanism.revolute(origin, crank_o, name="crank_input")
    mechanism.revolute(crank_b, rod_b)
    mechanism.revolute(rod_c, slider_c)
    prismatic_joint = mechanism.prismatic(
        guide,
        slider_guide,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="slider_guide",
    )

    initial_guess = {
        crank: (0.0, 0.0, 0.7),
        connecting_rod: (0.06, 0.05, -0.2),
        slider: (0.30, 0.0, 0.0),
    }
    return mechanism, crank_joint, prismatic_joint, initial_guess


def main():
    mechanism, crank_joint, prismatic_joint, initial_guess = build_mechanism()

    input_angle = np.linspace(0.7, 0.7 + 2 * np.pi, 181)
    input_angular_velocity = 1.0
    input_angular_acceleration = 0.0

    solution = solve(
        mechanism,
        input=crank_joint,
        values=input_angle,
        input_velocity=input_angular_velocity,
        input_acceleration=input_angular_acceleration,
        initial_guess=initial_guess,
    )

    slider_position = solution.joint_coordinates(prismatic_joint)
    slider_velocity = solution.joint_velocities(prismatic_joint)
    slider_acceleration = solution.joint_accelerations(prismatic_joint)

    print("Slider-crank differential analysis")
    print(f"Solved {len(solution)} configurations")
    print(
        "Slider displacement range: "
        f"{slider_position.min():.6f} -> {slider_position.max():.6f}"
    )
    print(
        "Slider velocity range: "
        f"{slider_velocity.min():.6f} -> {slider_velocity.max():.6f}"
    )
    print(
        "Slider acceleration range: "
        f"{slider_acceleration.min():.6f} -> {slider_acceleration.max():.6f}"
    )

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(input_angle, slider_position)
    axes[0].set_ylabel("displacement")
    axes[0].set_title("Slider kinematics")

    axes[1].plot(input_angle, slider_velocity)
    axes[1].set_ylabel("velocity")

    axes[2].plot(input_angle, slider_acceleration)
    axes[2].set_xlabel("input crank angle [rad]")
    axes[2].set_ylabel("acceleration")

    fig.tight_layout()

    # The same kinematic state can be reconstructed with the prismatic
    # coordinate prescribed instead of the crank angle.
    sample = 30
    reference = solution[sample]
    reconstructed = solve(
        mechanism,
        input=prismatic_joint,
        values=slider_position[sample],
        input_velocity=slider_velocity[sample],
        input_acceleration=slider_acceleration[sample],
        initial_guess=reference,
    )

    coordinate_error = np.max(
        np.abs(reconstructed.coordinates - reference.coordinates)
    )
    velocity_error = np.max(
        np.abs(reconstructed.coordinate_velocities - reference.coordinate_velocities)
    )
    acceleration_error = np.max(
        np.abs(
            reconstructed.coordinate_accelerations
            - reference.coordinate_accelerations
        )
    )

    print("Prismatic-input reconstruction")
    print(f"Maximum coordinate error: {coordinate_error:.3e}")
    print(f"Maximum velocity error: {velocity_error:.3e}")
    print(f"Maximum acceleration error: {acceleration_error:.3e}")

    plt.show()


if __name__ == "__main__":
    main()
