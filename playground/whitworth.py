"""Explore a Whitworth quick-return mechanism with Kimech's public API."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate


CRANK_RADIUS = 0.06
PIVOT_DISTANCE = 0.20
LEVER_ARM = 0.25
CONNECTING_ROD_LENGTH = 0.28


def build_mechanism():
    """Build a Whitworth quick-return mechanism and an initial pose estimate."""
    mechanism = Mechanism("whitworth")
    ground = mechanism.ground

    crank_center = ground.add_point("O", (0.0, 0.0))
    lever_pivot = ground.add_point("A", (PIVOT_DISTANCE, 0.0))
    ram_guide = ground.add_point("G", (0.0, 0.0))

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_b = crank.add_point("B", (CRANK_RADIUS, 0.0))

    block = mechanism.add_link("sliding_block")
    block_b = block.add_point("B", (0.0, 0.0))
    block_slot = block.add_point("S", (0.0, 0.0))

    lever = mechanism.add_link("slotted_lever")
    lever_a = lever.add_point("A", (0.0, 0.0))
    lever_slot = lever.add_point("S", (0.0, 0.0))
    lever_c = lever.add_point("C", (-LEVER_ARM, 0.0))

    connecting_rod = mechanism.add_link("connecting_rod")
    rod_c = connecting_rod.add_point("C", (0.0, 0.0))
    rod_r = connecting_rod.add_point("R", (CONNECTING_ROD_LENGTH, 0.0))

    ram = mechanism.add_link("ram")
    ram_r = ram.add_point("R", (0.0, 0.0))
    ram_g = ram.add_point("G", (0.0, 0.0))

    input_joint = mechanism.revolute(crank_center, crank_o, name="crank_input")
    mechanism.revolute(crank_b, block_b, name="crank_pin")
    slot_joint = mechanism.prismatic(
        lever_slot,
        block_slot,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="slot",
    )
    mechanism.revolute(lever_pivot, lever_a, name="lever_pivot")
    mechanism.revolute(lever_c, rod_c, name="lever_output")
    mechanism.revolute(rod_r, ram_r, name="ram_pin")
    ram_joint = mechanism.prismatic(
        ram_guide,
        ram_g,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="ram_guide",
    )

    theta0 = 0.55
    crank_pin = np.array(
        [
            CRANK_RADIUS * np.cos(theta0),
            CRANK_RADIUS * np.sin(theta0),
        ]
    )

    lever_angle = np.arctan2(
        crank_pin[1],
        crank_pin[0] - PIVOT_DISTANCE,
    )
    lever_tip = np.array(
        [
            PIVOT_DISTANCE - LEVER_ARM * np.cos(lever_angle),
            -LEVER_ARM * np.sin(lever_angle),
        ]
    )

    horizontal_reach = np.sqrt(
        CONNECTING_ROD_LENGTH**2 - lever_tip[1] ** 2
    )
    ram_x = lever_tip[0] + horizontal_reach
    rod_angle = np.arctan2(-lever_tip[1], horizontal_reach)

    initial_guess = {
        crank: (0.0, 0.0, theta0),
        block: (crank_pin[0], crank_pin[1], lever_angle),
        lever: (PIVOT_DISTANCE, 0.0, lever_angle),
        connecting_rod: (lever_tip[0], lever_tip[1], rod_angle),
        ram: (ram_x, 0.0, 0.0),
    }

    return mechanism, input_joint, slot_joint, ram_joint, initial_guess, theta0


def quick_return_spans(values, positions):
    """Return the two crank-angle spans between the ram stroke endpoints."""
    minimum_index = int(np.argmin(positions))
    maximum_index = int(np.argmax(positions))

    first_span = (
        values[maximum_index] - values[minimum_index]
    ) % (2.0 * np.pi)
    second_span = 2.0 * np.pi - first_span
    return first_span, second_span


def main():
    (
        mechanism,
        input_joint,
        slot_joint,
        ram_joint,
        initial_guess,
        theta0,
    ) = build_mechanism()

    report = mechanism.validate()
    values = np.linspace(
        theta0,
        theta0 + 2.0 * np.pi,
        360,
        endpoint=False,
    )

    solution = solve(
        mechanism,
        input=input_joint,
        values=values,
        initial_guess=initial_guess,
    )

    slot_positions = solution.joint_coordinates(slot_joint)
    expected_slot_positions = np.sqrt(
        PIVOT_DISTANCE**2
        + CRANK_RADIUS**2
        - 2.0 * PIVOT_DISTANCE * CRANK_RADIUS * np.cos(values)
    )
    slot_error = np.max(np.abs(slot_positions - expected_slot_positions))

    ram_positions = solution.joint_coordinates(ram_joint)
    span_a, span_b = quick_return_spans(values, ram_positions)
    long_span = max(span_a, span_b)
    short_span = min(span_a, span_b)
    quick_return_ratio = long_span / short_span

    print("Whitworth quick-return mechanism")
    print(f"Valid model: {report.is_valid}")
    print(f"Mobility: {report.mobility}")
    print(f"Solved {len(solution)} configurations")
    print(f"Maximum moving-slot error: {slot_error:.3e}")
    print(
        "Ram stroke: "
        f"{ram_positions.min():.6f} -> {ram_positions.max():.6f}"
    )
    print(
        "Crank-angle spans between stroke endpoints: "
        f"{np.degrees(long_span):.1f} deg / {np.degrees(short_span):.1f} deg"
    )
    print(f"Quick-return ratio: {quick_return_ratio:.3f}")

    fig, ax = plt.subplots()
    animation = animate(
        solution,
        fps=30,
        ax=ax,
    )

    ax.set_title("Whitworth quick-return mechanism")
    plt.show()

    # Keep a reference alive until Matplotlib has finished displaying the animation.
    _ = animation


if __name__ == "__main__":
    main()
