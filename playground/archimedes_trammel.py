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

    mechanism.prismatic(
        horizontal_guide,
        horizontal_g,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="horizontal_guide",
    )
    mechanism.prismatic(
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

    return mechanism, input_joint, tracer, initial_guess, theta0


def main():
    mechanism, input_joint, tracer, initial_guess, theta0 = build_mechanism()
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

    path = solution.point_path(tracer)

    expected_a = BAR_LENGTH - TRACER_DISTANCE
    expected_b = TRACER_DISTANCE
    ellipse_residual = (path[:, 0] / expected_a) ** 2 + (path[:, 1] / expected_b) ** 2 - 1.0

    print("Archimedes trammel")
    print(f"Solved {len(solution)} configurations")
    print(f"Expected ellipse semiaxes: a={expected_a:.3f}, b={expected_b:.3f}")
    print(f"Maximum ellipse residual: {np.max(np.abs(ellipse_residual)):.3e}")

    fig, ax = plt.subplots()
    ax.plot(path[:, 0], path[:, 1], "--", linewidth=1.0, label="Tracer path")

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
