"""Explore a Watt II six-bar linkage with Kimech's public API."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate


CRANK_RADIUS = 0.07
GROUND_O2_O4 = 0.28
INPUT_ROD_LENGTH = 0.18
TERNARY_B = np.array((0.20, 0.0))
TERNARY_C = np.array((0.12, -0.10))
TRACER_LOCAL = np.array((0.08, 0.06))
GROUND_O6 = np.array((0.325, 0.225))
SECOND_ROD_LENGTH = 0.12
OUTPUT_ROCKER_LENGTH = 0.08


def circle_intersections(center_a, radius_a, center_b, radius_b):
    """Return the two intersections of two non-degenerate circles."""
    delta = center_b - center_a
    distance = np.linalg.norm(delta)
    along = (
        radius_a**2 - radius_b**2 + distance**2
    ) / (2.0 * distance)
    height = np.sqrt(max(0.0, radius_a**2 - along**2))

    direction = delta / distance
    normal = np.array((-direction[1], direction[0]))
    base = center_a + along * direction
    return base + height * normal, base - height * normal


def analytical_geometry(values):
    """Construct the selected assembly branch from circle intersections."""
    o4 = np.array((GROUND_O2_O4, 0.0))

    points_b = []
    points_c = []
    points_d = []
    tracers = []
    previous_b = None
    previous_d = None

    for theta in values:
        point_a = CRANK_RADIUS * np.array((np.cos(theta), np.sin(theta)))

        candidates_b = circle_intersections(
            point_a,
            INPUT_ROD_LENGTH,
            o4,
            np.linalg.norm(TERNARY_B),
        )
        if previous_b is None:
            point_b = max(candidates_b, key=lambda point: point[1])
        else:
            point_b = min(
                candidates_b,
                key=lambda point: np.linalg.norm(point - previous_b),
            )
        previous_b = point_b

        ternary_angle = np.arctan2(
            point_b[1] - o4[1],
            point_b[0] - o4[0],
        )
        cosine = np.cos(ternary_angle)
        sine = np.sin(ternary_angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))

        point_c = o4 + rotation @ TERNARY_C
        tracer = o4 + rotation @ TRACER_LOCAL

        candidates_d = circle_intersections(
            point_c,
            SECOND_ROD_LENGTH,
            GROUND_O6,
            OUTPUT_ROCKER_LENGTH,
        )
        if previous_d is None:
            point_d = min(candidates_d, key=lambda point: point[1])
        else:
            point_d = min(
                candidates_d,
                key=lambda point: np.linalg.norm(point - previous_d),
            )
        previous_d = point_d

        points_b.append(point_b)
        points_c.append(point_c)
        points_d.append(point_d)
        tracers.append(tracer)

    return (
        np.array(points_b),
        np.array(points_c),
        np.array(points_d),
        np.array(tracers),
    )


def build_mechanism():
    """Build a Watt II six-bar linkage and a suitable initial pose estimate."""
    mechanism = Mechanism("watt_ii_six_bar")
    ground = mechanism.ground

    ground_o2 = ground.add_point("O2", (0.0, 0.0))
    ground_o4 = ground.add_point("O4", (GROUND_O2_O4, 0.0))
    ground_o6 = ground.add_point("O6", GROUND_O6)

    crank = mechanism.add_link("crank")
    crank_o2 = crank.add_point("O2", (0.0, 0.0))
    crank_a = crank.add_point("A", (CRANK_RADIUS, 0.0))

    input_rod = mechanism.add_link("input_rod")
    rod_a = input_rod.add_point("A", (0.0, 0.0))
    rod_b = input_rod.add_point("B", (INPUT_ROD_LENGTH, 0.0))

    ternary = mechanism.add_link("ternary_rocker")
    ternary_o4 = ternary.add_point("O4", (0.0, 0.0))
    ternary_b = ternary.add_point("B", TERNARY_B)
    ternary_c = ternary.add_point("C", TERNARY_C)
    tracer = ternary.add_point("P", TRACER_LOCAL)

    second_rod = mechanism.add_link("second_rod")
    second_c = second_rod.add_point("C", (0.0, 0.0))
    second_d = second_rod.add_point("D", (SECOND_ROD_LENGTH, 0.0))

    output_rocker = mechanism.add_link("output_rocker")
    output_d = output_rocker.add_point("D", (0.0, 0.0))
    output_o6 = output_rocker.add_point("O6", (OUTPUT_ROCKER_LENGTH, 0.0))

    input_joint = mechanism.revolute(ground_o2, crank_o2, name="crank_input")
    mechanism.revolute(crank_a, rod_a, name="joint_a")
    mechanism.revolute(rod_b, ternary_b, name="joint_b")
    mechanism.revolute(ternary_o4, ground_o4, name="ternary_ground")
    mechanism.revolute(ternary_c, second_c, name="joint_c")
    mechanism.revolute(second_d, output_d, name="joint_d")
    mechanism.revolute(output_o6, ground_o6, name="output_ground")

    theta0 = 0.60
    expected_b, expected_c, expected_d, _ = analytical_geometry(
        np.array((theta0,))
    )

    point_a = CRANK_RADIUS * np.array((np.cos(theta0), np.sin(theta0)))
    point_b = expected_b[0]
    point_c = expected_c[0]
    point_d = expected_d[0]

    input_rod_angle = np.arctan2(
        point_b[1] - point_a[1],
        point_b[0] - point_a[0],
    )
    ternary_angle = np.arctan2(
        point_b[1],
        point_b[0] - GROUND_O2_O4,
    )
    second_rod_angle = np.arctan2(
        point_d[1] - point_c[1],
        point_d[0] - point_c[0],
    )
    output_angle = np.arctan2(
        GROUND_O6[1] - point_d[1],
        GROUND_O6[0] - point_d[0],
    )

    initial_guess = {
        crank: (0.0, 0.0, theta0),
        input_rod: (point_a[0], point_a[1], input_rod_angle),
        ternary: (GROUND_O2_O4, 0.0, ternary_angle),
        second_rod: (point_c[0], point_c[1], second_rod_angle),
        output_rocker: (point_d[0], point_d[1], output_angle),
    }

    return (
        mechanism,
        input_joint,
        ternary_b,
        ternary_c,
        tracer,
        output_rocker,
        initial_guess,
        theta0,
    )


def main():
    (
        mechanism,
        input_joint,
        point_b,
        point_c,
        tracer,
        output_rocker,
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

    expected_b, expected_c, expected_d, expected_tracer = analytical_geometry(values)
    solved_b = solution.point_path(point_b)
    solved_c = solution.point_path(point_c)
    solved_d = solution.point_path(output_rocker["D"])
    solved_tracer = solution.point_path(tracer)

    b_error = np.max(np.linalg.norm(solved_b - expected_b, axis=1))
    c_error = np.max(np.linalg.norm(solved_c - expected_c, axis=1))
    d_error = np.max(np.linalg.norm(solved_d - expected_d, axis=1))
    tracer_error = np.max(
        np.linalg.norm(solved_tracer - expected_tracer, axis=1)
    )

    output_angles = np.unwrap(solution.link_poses(output_rocker)[:, 2])
    output_swing = np.degrees(output_angles.max() - output_angles.min())

    print("Watt II six-bar linkage")
    print(f"Valid model: {report.is_valid}")
    print(f"Mobility: {report.mobility}")
    print(f"Mobile links: {len(mechanism.links)}")
    print(f"Revolute joints: {len(mechanism.joints)}")
    print(f"Solved {len(solution)} configurations")
    print(f"Maximum point-B error: {b_error:.3e}")
    print(f"Maximum point-C error: {c_error:.3e}")
    print(f"Maximum point-D error: {d_error:.3e}")
    print(f"Maximum tracer-path error: {tracer_error:.3e}")
    print(f"Output-rocker angular swing: {output_swing:.2f} deg")

    fig, ax = plt.subplots()
    ax.plot(
        solved_tracer[:, 0],
        solved_tracer[:, 1],
        "--",
        linewidth=1.0,
        label="Ternary-link point path",
    )

    animation = animate(
        solution,
        fps=30,
        ax=ax,
    )

    ax.set_title("Watt II six-bar linkage")
    ax.legend()
    plt.show()

    # Keep a reference alive until Matplotlib has finished displaying the animation.
    _ = animation


if __name__ == "__main__":
    main()
