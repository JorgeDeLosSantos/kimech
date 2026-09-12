"""Solve and animate a slider-crank through Kimech's public API."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate


def build_mechanism():
    """Build a slider-crank and a suitable initial pose estimate."""
    mechanism = Mechanism("slider_crank")
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
    values = np.linspace(0.7, 0.7 + 2 * np.pi, 180, endpoint=False)

    solution = solve(
        mechanism,
        input=crank_joint,
        values=values,
        initial_guess=initial_guess,
    )

    slider_positions = solution.joint_coordinates(prismatic_joint)

    print("Slider-crank motion")
    print(f"Solved {len(solution)} configurations")
    print(f"Crank input: {solution.input_values[0]:.3f} -> {solution.input_values[-1]:.3f} rad")
    print(
        f"Slider displacement: {slider_positions.min():.6f} -> "
        f"{slider_positions.max():.6f}"
    )

    fig, ax = plt.subplots()
    animation = animate(solution, fps=30, ax=ax)
    ax.set_title("Slider-crank mechanism")
    plt.show()


if __name__ == "__main__":
    main()
