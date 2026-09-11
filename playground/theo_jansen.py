"""Explore a Theo Jansen leg with Kimech's public API.

The geometry uses the canonical Strandbeest "holy numbers" proportions.
A uniform scale is applied internally so the absolute solver tolerance is not
needlessly sensitive to the conventional numerical size of those proportions.
"""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate


# Theo Jansen's canonical "holy numbers". Their absolute scale is arbitrary;
# only the proportions matter. Keeping the model near unit scale makes the
# solver's absolute residual tolerance comparable to the other playground cases.
SCALE = 0.01
A = 38.0 * SCALE
B = 41.5 * SCALE
C = 39.3 * SCALE
D = 40.1 * SCALE
E = 55.8 * SCALE
F = 39.4 * SCALE
G = 36.7 * SCALE
H = 65.7 * SCALE
I = 49.0 * SCALE
J = 50.0 * SCALE
K = 61.9 * SCALE
L = 7.8 * SCALE
M = 15.0 * SCALE

GROUND_O2 = np.array((0.0, 0.0))
GROUND_O4 = np.array((A, L))


def rotation_matrix(angle):
    """Return a planar rotation matrix."""
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(((cosine, -sine), (sine, cosine)))


def circle_intersections(center_a, radius_a, center_b, radius_b):
    """Return the two intersections of two non-degenerate circles."""
    delta = center_b - center_a
    distance = np.linalg.norm(delta)
    along = (
        radius_a**2 - radius_b**2 + distance**2
    ) / (2.0 * distance)
    height_squared = radius_a**2 - along**2
    height = np.sqrt(max(0.0, height_squared))

    direction = delta / distance
    normal = np.array((-direction[1], direction[0]))
    base = center_a + along * direction
    return base + height * normal, base - height * normal


UPPER_D_LOCAL = circle_intersections(
    np.array((0.0, 0.0)),
    D,
    np.array((B, 0.0)),
    E,
)[0]

FOOT_LOCAL = circle_intersections(
    np.array((0.0, 0.0)),
    I,
    np.array((G, 0.0)),
    H,
)[0]


def analytical_geometry(values):
    """Construct the canonical assembly branch from circle intersections."""
    crank_pins = []
    upper_joints = []
    lower_joints = []
    upper_side_joints = []
    connector_joints = []
    feet = []

    previous_upper = None
    previous_lower = None
    previous_connector = None

    for theta in values:
        crank_pin = GROUND_O4 + M * np.array(
            (np.cos(theta), np.sin(theta))
        )

        upper_candidates = circle_intersections(
            GROUND_O2,
            B,
            crank_pin,
            J,
        )
        lower_candidates = circle_intersections(
            GROUND_O2,
            C,
            crank_pin,
            K,
        )

        if previous_upper is None:
            upper_joint = max(upper_candidates, key=lambda point: point[1])
            lower_joint = min(lower_candidates, key=lambda point: point[1])
        else:
            upper_joint = min(
                upper_candidates,
                key=lambda point: np.linalg.norm(point - previous_upper),
            )
            lower_joint = min(
                lower_candidates,
                key=lambda point: np.linalg.norm(point - previous_lower),
            )

        previous_upper = upper_joint
        previous_lower = lower_joint

        upper_angle = np.arctan2(upper_joint[1], upper_joint[0])
        upper_side_joint = rotation_matrix(upper_angle) @ UPPER_D_LOCAL

        connector_candidates = circle_intersections(
            upper_side_joint,
            F,
            lower_joint,
            G,
        )
        if previous_connector is None:
            connector_joint = min(
                connector_candidates,
                key=lambda point: point[0],
            )
        else:
            connector_joint = min(
                connector_candidates,
                key=lambda point: np.linalg.norm(
                    point - previous_connector
                ),
            )
        previous_connector = connector_joint

        foot_angle = np.arctan2(
            connector_joint[1] - lower_joint[1],
            connector_joint[0] - lower_joint[0],
        )
        foot = lower_joint + rotation_matrix(foot_angle) @ FOOT_LOCAL

        crank_pins.append(crank_pin)
        upper_joints.append(upper_joint)
        lower_joints.append(lower_joint)
        upper_side_joints.append(upper_side_joint)
        connector_joints.append(connector_joint)
        feet.append(foot)

    return (
        np.array(crank_pins),
        np.array(upper_joints),
        np.array(lower_joints),
        np.array(upper_side_joints),
        np.array(connector_joints),
        np.array(feet),
    )


def build_mechanism():
    """Build a canonical Theo Jansen leg and an initial pose estimate."""
    mechanism = Mechanism("theo_jansen")
    ground = mechanism.ground

    ground_o2 = ground.add_point("O2", GROUND_O2)
    ground_o4 = ground.add_point("O4", GROUND_O4)

    crank = mechanism.add_link("crank")
    crank_o4 = crank.add_point("O4", (0.0, 0.0))
    crank_p = crank.add_point("P", (M, 0.0))

    upper_coupler = mechanism.add_link("upper_coupler_j")
    upper_coupler_p = upper_coupler.add_point("P", (0.0, 0.0))
    upper_coupler_u = upper_coupler.add_point("U", (J, 0.0))

    upper_ternary = mechanism.add_link("upper_ternary_bde")
    upper_o2 = upper_ternary.add_point("O2", (0.0, 0.0))
    upper_u = upper_ternary.add_point("U", (B, 0.0))
    upper_d = upper_ternary.add_point("D", UPPER_D_LOCAL)

    lower_coupler = mechanism.add_link("lower_coupler_k")
    lower_coupler_p = lower_coupler.add_point("P", (0.0, 0.0))
    lower_coupler_l = lower_coupler.add_point("L", (K, 0.0))

    lower_rocker = mechanism.add_link("lower_rocker_c")
    lower_o2 = lower_rocker.add_point("O2", (0.0, 0.0))
    lower_l = lower_rocker.add_point("L", (C, 0.0))

    connector = mechanism.add_link("connector_f")
    connector_d = connector.add_point("D", (0.0, 0.0))
    connector_s = connector.add_point("S", (F, 0.0))

    foot_ternary = mechanism.add_link("foot_ternary_ghi")
    foot_l = foot_ternary.add_point("L", (0.0, 0.0))
    foot_s = foot_ternary.add_point("S", (G, 0.0))
    foot = foot_ternary.add_point("F", FOOT_LOCAL)

    input_joint = mechanism.revolute(
        ground_o4,
        crank_o4,
        name="crank_input",
    )
    mechanism.revolute(crank_p, upper_coupler_p, name="crank_upper")
    mechanism.revolute(upper_coupler_u, upper_u, name="upper_joint")
    mechanism.revolute(upper_o2, ground_o2, name="upper_ground")

    mechanism.revolute(crank_p, lower_coupler_p, name="crank_lower")
    mechanism.revolute(lower_coupler_l, lower_l, name="lower_joint")
    mechanism.revolute(lower_o2, ground_o2, name="lower_ground")

    mechanism.revolute(upper_d, connector_d, name="side_joint")
    mechanism.revolute(connector_s, foot_s, name="connector_joint")
    mechanism.revolute(lower_l, foot_l, name="foot_joint")

    theta0 = 0.0
    (
        expected_p,
        expected_u,
        expected_l,
        expected_d,
        expected_s,
        _,
    ) = analytical_geometry(np.array((theta0,)))

    point_p = expected_p[0]
    point_u = expected_u[0]
    point_l = expected_l[0]
    point_d = expected_d[0]
    point_s = expected_s[0]

    initial_guess = {
        crank: (GROUND_O4[0], GROUND_O4[1], theta0),
        upper_coupler: (
            point_p[0],
            point_p[1],
            np.arctan2(
                point_u[1] - point_p[1],
                point_u[0] - point_p[0],
            ),
        ),
        upper_ternary: (
            0.0,
            0.0,
            np.arctan2(point_u[1], point_u[0]),
        ),
        lower_coupler: (
            point_p[0],
            point_p[1],
            np.arctan2(
                point_l[1] - point_p[1],
                point_l[0] - point_p[0],
            ),
        ),
        lower_rocker: (
            0.0,
            0.0,
            np.arctan2(point_l[1], point_l[0]),
        ),
        connector: (
            point_d[0],
            point_d[1],
            np.arctan2(
                point_s[1] - point_d[1],
                point_s[0] - point_d[0],
            ),
        ),
        foot_ternary: (
            point_l[0],
            point_l[1],
            np.arctan2(
                point_s[1] - point_l[1],
                point_s[0] - point_l[0],
            ),
        ),
    }

    return (
        mechanism,
        input_joint,
        upper_u,
        lower_l,
        upper_d,
        connector_s,
        foot,
        initial_guess,
        theta0,
    )


def main():
    (
        mechanism,
        input_joint,
        upper_joint,
        lower_joint,
        side_joint,
        connector_joint,
        foot,
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

    (
        _,
        expected_upper,
        expected_lower,
        expected_side,
        expected_connector,
        expected_foot,
    ) = analytical_geometry(values)

    solved_upper = solution.point_path(upper_joint)
    solved_lower = solution.point_path(lower_joint)
    solved_side = solution.point_path(side_joint)
    solved_connector = solution.point_path(connector_joint)
    solved_foot = solution.point_path(foot)

    upper_error = np.max(
        np.linalg.norm(solved_upper - expected_upper, axis=1)
    )
    lower_error = np.max(
        np.linalg.norm(solved_lower - expected_lower, axis=1)
    )
    side_error = np.max(
        np.linalg.norm(solved_side - expected_side, axis=1)
    )
    connector_error = np.max(
        np.linalg.norm(solved_connector - expected_connector, axis=1)
    )
    foot_error = np.max(
        np.linalg.norm(solved_foot - expected_foot, axis=1)
    )

    foot_span = np.ptp(solved_foot, axis=0)
    minimum_height = solved_foot[:, 1].min()
    stance_mask = solved_foot[:, 1] <= (
        minimum_height + 0.15 * foot_span[1]
    )
    stance_span = np.ptp(solved_foot[stance_mask], axis=0)

    # Convert geometric spans back to the conventional holy-number scale for
    # easier comparison with published Jansen dimensions.
    reported_foot_span = foot_span / SCALE
    reported_stance_span = stance_span / SCALE

    print("Theo Jansen linkage")
    print("Reference geometry: canonical Strandbeest holy numbers")
    print(f"Internal geometry scale: {SCALE:g}")
    print(f"Valid model: {report.is_valid}")
    print(f"Mobility: {report.mobility}")
    print(f"Mobile links: {len(mechanism.links)}")
    print(f"Revolute joints: {len(mechanism.joints)}")
    print(f"Solved {len(solution)} configurations")
    print(f"Maximum upper-joint error: {upper_error:.3e}")
    print(f"Maximum lower-joint error: {lower_error:.3e}")
    print(f"Maximum side-joint error: {side_error:.3e}")
    print(f"Maximum connector-joint error: {connector_error:.3e}")
    print(f"Maximum foot-path error: {foot_error:.3e}")
    print(
        "Foot-path span (holy-number units): "
        f"dx={reported_foot_span[0]:.3f}, "
        f"dy={reported_foot_span[1]:.3f}"
    )
    print(
        "Approx. stance span (lowest 15% of foot height, holy-number units): "
        f"dx={reported_stance_span[0]:.3f}, "
        f"dy={reported_stance_span[1]:.3f}"
    )

    fig, ax = plt.subplots()
    ax.plot(
        solved_foot[:, 0],
        solved_foot[:, 1],
        "--",
        linewidth=1.0,
        label="Foot path",
    )

    animation = animate(
        solution,
        fps=30,
        ax=ax,
    )

    ax.set_title("Theo Jansen linkage")
    ax.legend()
    plt.show()

    # Keep a reference alive until Matplotlib has finished displaying the animation.
    _ = animation


if __name__ == "__main__":
    main()
