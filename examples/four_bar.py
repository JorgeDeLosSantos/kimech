"""Solve a four-bar linkage through Kimech's public API."""

import numpy as np

from kimech import Mechanism, solve


def build_mechanism():
    """Build a four-bar linkage and a suitable initial pose estimate."""
    mechanism = Mechanism("four_bar")
    ground = mechanism.ground

    ground_a = ground.add_point("A", (0.0, 0.0))
    ground_d = ground.add_point("D", (0.30, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22, 0.0))
    coupler_point = coupler.add_point("P", (0.10, 0.05))

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)

    # Each entry estimates a link pose as (x, y, theta).
    initial_guess = {
        crank: (0.0, 0.0, 0.8),
        coupler: (0.05, 0.06, 0.2),
        rocker: (0.30, 0.0, 2.2),
    }
    return mechanism, input_joint, crank, rocker, coupler_point, initial_guess


def main():
    mechanism, input_joint, crank, rocker, point_p, initial_guess = build_mechanism()
    values = np.linspace(0.8, 1.3, 100)

    solution = solve(
        mechanism,
        input=input_joint,
        values=values,
        initial_guess=initial_guess,
    )

    path = solution.point_path(point_p)
    rocker_poses = solution.link_poses(rocker)
    input_coordinates = solution.joint_coordinates(input_joint)

    first_config = solution[0]
    first_point_position = first_config.position(point_p)
    first_crank_pose = first_config.pose(crank)

    print("Four-bar")
    print(f"Solved {len(solution)} configurations")
    print(f"Input: {input_coordinates[0]:.3f} -> {input_coordinates[-1]:.3f} rad")
    print(f"Coupler point P: {first_point_position} -> {path[-1]}")
    print(f"First crank pose: {first_crank_pose}")
    print(f"Rocker angle: {rocker_poses[0, 2]:.3f} -> {rocker_poses[-1, 2]:.3f} rad")


if __name__ == "__main__":
    main()