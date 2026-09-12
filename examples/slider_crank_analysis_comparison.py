"""Compare Kimech slider-crank kinematics against the analytical solution."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve


def build_mechanism():
    """Build the slider-crank used in the analytical comparison."""
    mechanism = Mechanism("slider_crank_analysis_comparison")
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


def analytical_slider_kinematics(
    theta,
    *,
    crank_length,
    rod_length,
    angular_velocity,
    angular_acceleration,
):
    """Return analytical slider position, velocity, and acceleration."""
    theta = np.asarray(theta, dtype=float)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    root = np.sqrt(rod_length**2 - crank_length**2 * sin_theta**2)

    position = crank_length * cos_theta + root

    dx_dtheta = (
        -crank_length * sin_theta
        - crank_length**2 * sin_theta * cos_theta / root
    )

    d2x_dtheta2 = (
        -crank_length * cos_theta
        - crank_length**2 * (cos_theta**2 - sin_theta**2) / root
        - crank_length**4 * sin_theta**2 * cos_theta**2 / root**3
    )

    velocity = dx_dtheta * angular_velocity
    acceleration = (
        d2x_dtheta2 * angular_velocity**2
        + dx_dtheta * angular_acceleration
    )
    return position, velocity, acceleration


def main():
    mechanism, crank_joint, prismatic_joint, initial_guess = build_mechanism()

    crank_length = 0.08
    rod_length = 0.24
    input_angle = np.linspace(0.7, 0.7 + 2 * np.pi, 181)
    input_angular_velocity = 1.2
    input_angular_acceleration = -0.25

    solution = solve(
        mechanism,
        input=crank_joint,
        values=input_angle,
        input_velocity=input_angular_velocity,
        input_acceleration=input_angular_acceleration,
        initial_guess=initial_guess,
    )

    kimech_position = solution.joint_coordinates(prismatic_joint)
    kimech_velocity = solution.joint_velocities(prismatic_joint)
    kimech_acceleration = solution.joint_accelerations(prismatic_joint)

    analytical_position, analytical_velocity, analytical_acceleration = (
        analytical_slider_kinematics(
            input_angle,
            crank_length=crank_length,
            rod_length=rod_length,
            angular_velocity=input_angular_velocity,
            angular_acceleration=input_angular_acceleration,
        )
    )

    position_error = np.max(np.abs(kimech_position - analytical_position))
    velocity_error = np.max(np.abs(kimech_velocity - analytical_velocity))
    acceleration_error = np.max(
        np.abs(kimech_acceleration - analytical_acceleration)
    )

    print("Slider-crank analytical comparison")
    print(f"Maximum position error: {position_error:.3e}")
    print(f"Maximum velocity error: {velocity_error:.3e}")
    print(f"Maximum acceleration error: {acceleration_error:.3e}")

    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(input_angle, kimech_position, label="Kimech")
    axes[0].plot(input_angle, analytical_position, "--", label="Analytical")
    axes[0].set_ylabel("position")
    axes[0].set_title("Slider-crank analytical comparison")
    axes[0].legend()

    axes[1].plot(input_angle, kimech_velocity, label="Kimech")
    axes[1].plot(input_angle, analytical_velocity, "--", label="Analytical")
    axes[1].set_ylabel("velocity")

    axes[2].plot(input_angle, kimech_acceleration, label="Kimech")
    axes[2].plot(input_angle, analytical_acceleration, "--", label="Analytical")
    axes[2].set_xlabel("input crank angle [rad]")
    axes[2].set_ylabel("acceleration")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
