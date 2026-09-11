"""Explore a Klann linkage with Kimech's public API.

The reference geometry is derived from Table 1 of Joseph C. Klann's
US Patent 6,260,862, using the fully extended stride position.
"""

import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate


# Published reference coordinates from US Patent 6,260,862, Table 1.
UPPER_PIVOT = np.array((1.366, 1.366))  # point 9
LOWER_PIVOT = np.array((1.009, 0.574))  # point 11
CRANK_CENTER = np.array((1.599, 0.750))  # point 15

ELBOW_0 = np.array((0.741, 0.750))  # point 27X
CRANK_PIN_0 = np.array((1.331, 0.750))  # point 29x
FOOT_0 = np.array((0.000, 0.000))  # point 33x
KNEE_0 = np.array((0.232, 0.866))  # point 35x
HIP_0 = np.array((0.866, 1.500))  # point 37x

CRANK_RADIUS = np.linalg.norm(CRANK_PIN_0 - CRANK_CENTER)
CONNECTING_TO_ELBOW = ELBOW_0 - CRANK_PIN_0
CONNECTING_TO_KNEE = KNEE_0 - CRANK_PIN_0
LOWER_ROCKER_LENGTH = np.linalg.norm(ELBOW_0 - LOWER_PIVOT)
LEG_TO_HIP = HIP_0 - KNEE_0
LEG_TO_FOOT = FOOT_0 - KNEE_0
UPPER_ROCKER_LENGTH = np.linalg.norm(HIP_0 - UPPER_PIVOT)


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


def analytical_geometry(values):
    """Construct the selected Klann assembly branch geometrically."""
    elbows = []
    knees = []
    hips = []
    feet = []

    previous_elbow = ELBOW_0
    previous_hip = HIP_0

    connecting_reference_angle = np.arctan2(
        CONNECTING_TO_ELBOW[1],
        CONNECTING_TO_ELBOW[0],
    )
    leg_reference_angle = np.arctan2(
        LEG_TO_HIP[1],
        LEG_TO_HIP[0],
    )

    connecting_elbow_radius = np.linalg.norm(CONNECTING_TO_ELBOW)
    leg_hip_radius = np.linalg.norm(LEG_TO_HIP)

    for theta in values:
        crank_pin = CRANK_CENTER + CRANK_RADIUS * np.array(
            (np.cos(theta), np.sin(theta))
        )

        elbow_candidates = circle_intersections(
            crank_pin,
            connecting_elbow_radius,
            LOWER_PIVOT,
            LOWER_ROCKER_LENGTH,
        )
        elbow = min(
            elbow_candidates,
            key=lambda point: np.linalg.norm(point - previous_elbow),
        )
        previous_elbow = elbow

        connecting_angle = (
            np.arctan2(
                elbow[1] - crank_pin[1],
                elbow[0] - crank_pin[0],
            )
            - connecting_reference_angle
        )
        knee = (
            crank_pin
            + rotation_matrix(connecting_angle) @ CONNECTING_TO_KNEE
        )

        hip_candidates = circle_intersections(
            knee,
            leg_hip_radius,
            UPPER_PIVOT,
            UPPER_ROCKER_LENGTH,
        )
        hip = min(
            hip_candidates,
            key=lambda point: np.linalg.norm(point - previous_hip),
        )
        previous_hip = hip

        leg_angle = (
            np.arctan2(
                hip[1] - knee[1],
                hip[0] - knee[0],
            )
            - leg_reference_angle
        )
        foot = knee + rotation_matrix(leg_angle) @ LEG_TO_FOOT

        elbows.append(elbow)
        knees.append(knee)
        hips.append(hip)
        feet.append(foot)

    return (
        np.array(elbows),
        np.array(knees),
        np.array(hips),
        np.array(feet),
    )


def build_mechanism():
    """Build the patent-reference Klann linkage and its initial pose."""
    mechanism = Mechanism("klann")
    ground = mechanism.ground

    ground_upper = ground.add_point("O_upper", UPPER_PIVOT)
    ground_lower = ground.add_point("O_lower", LOWER_PIVOT)
    ground_crank = ground.add_point("O_crank", CRANK_CENTER)

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_pin = crank.add_point("P", (CRANK_RADIUS, 0.0))

    connecting_rod = mechanism.add_link("connecting_rod")
    rod_pin = connecting_rod.add_point("P", (0.0, 0.0))
    rod_elbow = connecting_rod.add_point("E", CONNECTING_TO_ELBOW)
    rod_knee = connecting_rod.add_point("K", CONNECTING_TO_KNEE)

    lower_rocker = mechanism.add_link("lower_rocker")
    lower_elbow = lower_rocker.add_point("E", (0.0, 0.0))
    lower_o = lower_rocker.add_point("O_lower", LOWER_PIVOT - ELBOW_0)

    leg = mechanism.add_link("leg")
    leg_hip = leg.add_point("H", LEG_TO_HIP)
    leg_knee = leg.add_point("K", (0.0, 0.0))
    foot = leg.add_point("F", LEG_TO_FOOT)

    upper_rocker = mechanism.add_link("upper_rocker")
    upper_hip = upper_rocker.add_point("H", (0.0, 0.0))
    upper_o = upper_rocker.add_point("O_upper", UPPER_PIVOT - HIP_0)

    input_joint = mechanism.revolute(
        ground_crank,
        crank_o,
        name="crank_input",
    )
    mechanism.revolute(crank_pin, rod_pin, name="crank_pin")
    mechanism.revolute(rod_elbow, lower_elbow, name="elbow")
    mechanism.revolute(lower_o, ground_lower, name="lower_rocker_ground")
    mechanism.revolute(rod_knee, leg_knee, name="knee")
    mechanism.revolute(leg_hip, upper_hip, name="hip")
    mechanism.revolute(upper_o, ground_upper, name="upper_rocker_ground")

    theta0 = np.pi
    initial_guess = {
        crank: (CRANK_CENTER[0], CRANK_CENTER[1], theta0),
        connecting_rod: (CRANK_PIN_0[0], CRANK_PIN_0[1], 0.0),
        lower_rocker: (ELBOW_0[0], ELBOW_0[1], 0.0),
        leg: (KNEE_0[0], KNEE_0[1], 0.0),
        upper_rocker: (HIP_0[0], HIP_0[1], 0.0),
    }

    return (
        mechanism,
        input_joint,
        rod_elbow,
        rod_knee,
        leg_hip,
        foot,
        initial_guess,
        theta0,
    )


def main():
    (
        mechanism,
        input_joint,
        elbow,
        knee,
        hip,
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

    expected_elbow, expected_knee, expected_hip, expected_foot = (
        analytical_geometry(values)
    )
    solved_elbow = solution.point_path(elbow)
    solved_knee = solution.point_path(knee)
    solved_hip = solution.point_path(hip)
    solved_foot = solution.point_path(foot)

    elbow_error = np.max(
        np.linalg.norm(solved_elbow - expected_elbow, axis=1)
    )
    knee_error = np.max(
        np.linalg.norm(solved_knee - expected_knee, axis=1)
    )
    hip_error = np.max(
        np.linalg.norm(solved_hip - expected_hip, axis=1)
    )
    foot_error = np.max(
        np.linalg.norm(solved_foot - expected_foot, axis=1)
    )

    foot_span = np.ptp(solved_foot, axis=0)

    print("Klann linkage")
    print("Reference geometry: US Patent 6,260,862, Table 1")
    print(f"Valid model: {report.is_valid}")
    print(f"Mobility: {report.mobility}")
    print(f"Mobile links: {len(mechanism.links)}")
    print(f"Revolute joints: {len(mechanism.joints)}")
    print(f"Solved {len(solution)} configurations")
    print(f"Maximum elbow error: {elbow_error:.3e}")
    print(f"Maximum knee error: {knee_error:.3e}")
    print(f"Maximum hip error: {hip_error:.3e}")
    print(f"Maximum foot-path error: {foot_error:.3e}")
    print(
        "Foot-path span: "
        f"dx={foot_span[0]:.3f}, dy={foot_span[1]:.3f}"
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

    ax.set_title("Klann linkage")
    ax.legend()
    plt.show()

    # Keep a reference alive until Matplotlib has finished displaying the animation.
    _ = animation


if __name__ == "__main__":
    main()
