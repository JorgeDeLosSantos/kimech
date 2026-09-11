"""Explore an Archimedes trammel with Kimech's public API."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate


BAR_LENGTH = 0.30
TRACER_DISTANCE = 0.12


def build_mechanism():
    """Build an Archimedes trammel and a suitable initial pose estimate."""
    mechanism = Mechanism("archimedes_trammel")
    ground = mechanism.ground

    horizontal_guide = ground.add_point("Gx", (0.0, 0.0))
    vertical_guide = ground.add_point("Gy", (0.0, 0.0))

    horizontal_slider = mechanism.add_link("horizontal_slider")
    horizontal_g = horizontal_slider.add_point("G", (0.0, 0.0))
    horizontal_a = horizontal_slider.add_point("A", (0.0, 0.0))

    vertical_slider = mechanism.add_link("vertical_slider")
    vertical_g = vertical_slider.add_point("G", (0.0, 0.0))
    vertical_b = vertical_slider.add_point("B", (0.0, 0.0))

    bar = mechanism.add_link("bar")
    bar_a = bar.add_point("A", (0.0, 0.0))
    bar_b = bar.add_point("B", (BAR_LENGTH, 0.0))
    tracer = bar.add_point("P", (TRACER_DISTANCE, 0.0))

    horizontal_joint = mechanism.prismatic(
        horizontal_guide,
        horizontal_g,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="horizontal_guide",
    )
    vertical_joint = mechanism.prismatic(
        vertical_guide,
        vertical_g,
        axis_a=(0.0, 1.0),
        axis_b=(0.0, 1.0),
        name="vertical_guide",
    )
    input_joint = mechanism.revolute(
        horizontal_a,
        bar_a,
        name="bar_angle",
    )
    mechanism.revolute(vertical_b, bar_b)

    theta0 = 0.55
    initial_guess = {
        horizontal_slider: (-BAR_LENGTH * np.cos(theta0), 0.0, 0.0),
        vertical_slider: (0.0, BAR_LENGTH * np.sin(theta0), 0.0),
        bar: (-BAR_LENGTH * np.cos(theta0), 0.0, theta0),
    }

    return (
        mechanism,
        input_joint,
        horizontal_joint,
        vertical_joint,
        tracer,
        initial_guess,
        theta0,
    )


def main():
    (
        mechanism,
        input_joint,
        horizontal_joint,
        vertical_joint,
        tracer,
        initial_guess,
        theta0,
    ) = build_mechanism()

    report = mechanism.validate()

    values = np.linspace(
        theta0,
        theta0 + 2 * np.pi,
        240,
        endpoint=False,
    )

    solution = solve(
        mechanism,
        input=input_joint,
        values=values,
        initial_guess=initial_guess,
    )

    theta = solution.input_values
    path = solution.point_path(tracer)
    horizontal_positions = solution.joint_coordinates(horizontal_joint)
    vertical_positions = solution.joint_coordinates(vertical_joint)

    expected_a = BAR_LENGTH - TRACER_DISTANCE
    expected_b = TRACER_DISTANCE
    analytical_path = np.column_stack(
        (
            -expected_a * np.cos(theta),
            expected_b * np.sin(theta),
        )
    )
    analytical_horizontal = -BAR_LENGTH * np.cos(theta)
    analytical_vertical = BAR_LENGTH * np.sin(theta)

    tracer_error = np.linalg.norm(path - analytical_path, axis=1)
    horizontal_error = np.abs(horizontal_positions - analytical_horizontal)
    vertical_error = np.abs(vertical_positions - analytical_vertical)

    print("Archimedes trammel")
    print(f"Valid model: {report.is_valid}")
    print(f"Mobility: {report.mobility}")
    print(f"Solved {len(solution)} configurations")
    print(f"Expected ellipse semiaxes: a={expected_a:.3f}, b={expected_b:.3f}")
    print(f"Maximum tracer-path error: {tracer_error.max():.3e}")
    print(f"Maximum horizontal-slider error: {horizontal_error.max():.3e}")
    print(f"Maximum vertical-slider error: {vertical_error.max():.3e}")

    fig, ax = plt.subplots()
    ax.plot(
        analytical_path[:, 0],
        analytical_path[:, 1],
        ":",
        linewidth=2.0,
        label="Analytical ellipse",
    )
    ax.plot(
        path[:, 0],
        path[:, 1],
        "--",
        linewidth=1.0,
        label="Kimech tracer path",
    )

    animation = animate(
        solution,
        fps=30,
        ax=ax,
    )

    ax.set_title("Archimedes trammel")
    ax.legend()
    plt.show()

    # Keep a reference alive until Matplotlib has finished displaying the animation.
    _ = animation


if __name__ == "__main__":
    main()
