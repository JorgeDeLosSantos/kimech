"""Explore a Watt six-bar linkage with Kimech's public API."""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate


CRANK_RADIUS = 0.07
GROUND_O2_O4 = 0.28
COUPLER_AB = 0.18
ROCKER_LENGTH = 0.20
COUPLER_C = np.array((0.09, -0.10))
TRACER_LOCAL = np.array((0.11, -0.03))
GROUND_O6 = np.array((0.38, -0.12))
CONNECTING_ROD_LENGTH = 0.22
OUTPUT_ROCKER_LENGTH = 0.14


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

    points_c = []
    points_d = []
    tracers = []
    previous_b = None
    previous_d = None

    for theta in values:
        point_a = CRANK_RADIUS * np.array((np.cos(theta), np.sin(theta)))

        candidates_b = circle_intersections(
            point_a,
            COUPLER_AB,
            o4,
            ROCKER_LENGTH,
        )
        if previous_b is None:
            point_b = max(candidates_b, key=lambda point: point[1])
        else:
            point_b = min(
                candidates_b,
                key=lambda point: np.linalg.norm(point - previous_b),
            )
        previous_b = point_b

        coupler_angle = np.arctan2(
            point_b[1] - point_a[1],
            point_b[0] - point_a[0],
        )
        cosine = np.cos(coupler_angle)
        sine = np.sin(coupler_angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))

        point_c = point_a + rotation @ COUPLER_C
        tracer = point_a + rotation @ TRACER_LOCAL

        candidates_d = circle_intersections(
            point_c,
            CONNECTING_ROD_LENGTH,
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

        points_c.append(point_c)
        points_d.append(point_d)
        tracers.append(tracer)

    return np.array(points_c), np.array(points_d), np.array(tracers)


def build_mechanism():
    """Build a Watt six-bar linkage and a suitable initial pose estimate."""
    mechanism = Mechanism("watt_six_bar")
    ground = mechanism.ground

    ground_o2 = ground.add_point("O2", (0.0, 0.0))
    ground_o4 = ground.add_point("O4", (GROUND_O2_O4, 0.0))
    ground_o6 = ground.add_point("O6", GROUND_O6)

    crank = mechanism.add_link("crank")
    crank_o2 = crank.add_point("O2", (0.0, 0.0))
    crank_a = crank.add_point("A", (CRANK_RADIUS, 0.0))

    coupler = mechanism.add_link("ternary_coupler")
    coupler_a = coupler.add_point("A", (0.0, 0.0))
    coupler_b = coupler.add_point("B", (COUPLER_AB, 0.0))
    coupler_c = coupler.add_point("C", COUPLER_C)
    tracer = coupler.add_point("P", TRACER_LOCAL)

    rocker = mechanism.add_link("rocker")
    rocker_b = rocker.add_point("B", (0.0, 0.0))
    rocker_o4 = rocker.add_point("O4", (ROCKER_LENGTH, 0.0))

    connecting_rod = mechanism.add_link("connecting_rod")
    rod_c = connecting_rod.add_point("C", (0.0, 0.0))
    rod_d = connecting_rod.add_point("D", (CONNECTING_ROD_LENGTH, 0.0))

    output_rocker = mechanism.add_link("output_rocker")
    output_d = output_rocker.add_point("D", (0.0, 0.0))
    output_o6 = output_rocker.add_point("O6", (OUTPUT_ROCKER_LENGTH, 0.0))

    input_joint = mechanism.revolute(ground_o2, crank_o2, name="crank_input")
    mechanism.revolute(crank_a, coupler_a, name="joint_a")
    mechanism.revolute(coupler_b, rocker_b, name="joint_b")
    mechanism.revolute(rocker_o4, ground_o4, name="rocker_ground")
    mechanism.revolute(coupler_c, rod_c, name="joint_c")
    mechanism.revolute(rod_d, output_d, name="joint_d")
    mechanism.revolute(output_o6, ground_o6, name="output_ground")

    theta0 = 0.60
    expected_c, expected_d, _ = analytical_geometry(np.array((theta0,)))

    point_a = CRANK_RADIUS * np.array((np.cos(theta0), np.sin(theta0)))
    o4 = np.array((GROUND_O2_O4, 0.0))
    point_b = max(
        circle_intersections(point_a, COUPLER_AB, o4, ROCKER_LENGTH),
        key=lambda point: point[1],
    )
    coupler_angle = np.arctan2(
        point_b[1] - point_a[1],
        point_b[0] - point_a[0],
    )
    rocker_angle = np.arctan2(
        ground_o4.local[1] - point_b[1],
        ground_o4.local[0] - point_b[0],
    )

    point_c = expected_c[0]
    point_d = expected_d[0]
    rod_angle = np.arctan2(
        point_d[1] - point_c[1],
        point_d[0] - point_c[0],
    )
    output_angle = np.arctan2(
        GROUND_O6[1] - point_d[1],
        GROUND_O6[0] - point_d[0],
    )

    initial_guess = {
        crank: (0.0, 0.0, theta0),
        coupler: (point_a[0], point_a[1], coupler_angle),
        rocker: (point_b[0], point_b[1], rocker_angle),
        connecting_rod: (point_c[0], point_c[1], rod_angle),
        output_rocker: (point_d[0], point_d[1], output_angle),
    }

    return mechanism, input_joint, coupler_c, tracer, output_rocker, initial_guess, theta0


def main():
    (
        mechanism,
        input_joint,
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

    expected_c, expected_d, expected_tracer = analytical_geometry(values)
    solved_c = solution.point_path(point_c)
    solved_tracer = solution.point_path(tracer)

    output_d = output_rocker["D"]
    solved_d = solution.point_path(output_d)

    c_error = np.max(np.linalg.norm(solved_c - expected_c, axis=1))
    d_error = np.max(np.linalg.norm(solved_d - expected_d, axis=1))
    tracer_error = np.max(
        np.linalg.norm(solved_tracer - expected_tracer, axis=1)
    )

    output_angles = np.unwrap(solution.link_poses(output_rocker)[:, 2])
    output_swing = np.degrees(output_angles.max() - output_angles.min())

    print("Watt six-bar linkage")
    print(f"Valid model: {report.is_valid}")
    print(f"Mobility: {report.mobility}")
    print(f"Mobile links: {len(mechanism.links)}")
    print(f"Revolute joints: {len(mechanism.joints)}")
    print(f"Solved {len(solution)} configurations")
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
        label="Coupler-point path",
    )

    animation = animate(
        solution,
        fps=30,
        ax=ax,
    )

    ax.set_title("Watt six-bar linkage")
    ax.legend()
    plt.show()

    # Keep a reference alive until Matplotlib has finished displaying the animation.
    _ = animation


if __name__ == "__main__":
    main()
