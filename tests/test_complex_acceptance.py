import numpy as np

from kimech import Mechanism, KinematicDriver, solve


def _wrapped_angle_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _archimedes_trammel():
    bar_length = 0.30
    tracer_distance = 0.12

    mechanism = Mechanism("archimedes_trammel_acceptance")
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
    bar_b = bar.add_point("B", (bar_length, 0.0))
    tracer = bar.add_point("P", (tracer_distance, 0.0))

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
    input_joint = mechanism.revolute(horizontal_a, bar_a, name="bar_angle")
    mechanism.revolute(vertical_b, bar_b)

    theta0 = 0.55
    guess = {
        horizontal_slider: (-bar_length * np.cos(theta0), 0.0, 0.0),
        vertical_slider: (0.0, bar_length * np.sin(theta0), 0.0),
        bar: (-bar_length * np.cos(theta0), 0.0, theta0),
    }
    return (
        mechanism,
        input_joint,
        horizontal_joint,
        vertical_joint,
        tracer,
        guess,
        theta0,
        bar_length,
        tracer_distance,
    )


def _whitworth():
    crank_radius = 0.06
    pivot_distance = 0.20
    lever_arm = 0.25
    rod_length = 0.28

    mechanism = Mechanism("whitworth_acceptance")
    ground = mechanism.ground
    crank_center = ground.add_point("O", (0.0, 0.0))
    lever_pivot = ground.add_point("A", (pivot_distance, 0.0))
    ram_guide = ground.add_point("G", (pivot_distance, 0.0))

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_b = crank.add_point("B", (crank_radius, 0.0))

    block = mechanism.add_link("sliding_block")
    block_b = block.add_point("B", (0.0, 0.0))
    block_slot = block.add_point("S", (0.0, 0.0))

    lever = mechanism.add_link("slotted_lever")
    lever_a = lever.add_point("A", (0.0, 0.0))
    lever_slot = lever.add_point("S", (0.0, 0.0))
    lever_c = lever.add_point("C", (-lever_arm, 0.0))

    rod = mechanism.add_link("connecting_rod")
    rod_c = rod.add_point("C", (0.0, 0.0))
    rod_r = rod.add_point("R", (rod_length, 0.0))

    ram = mechanism.add_link("ram")
    ram_r = ram.add_point("R", (0.0, 0.0))
    ram_g = ram.add_point("G", (0.0, 0.0))

    input_joint = mechanism.revolute(crank_center, crank_o, name="crank_input")
    mechanism.revolute(crank_b, block_b)
    slot_joint = mechanism.prismatic(
        lever_slot,
        block_slot,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="slot",
    )
    mechanism.revolute(lever_pivot, lever_a)
    mechanism.revolute(lever_c, rod_c)
    mechanism.revolute(rod_r, ram_r)
    ram_joint = mechanism.prismatic(
        ram_guide,
        ram_g,
        axis_a=(0.0, 1.0),
        axis_b=(0.0, 1.0),
        name="ram_guide",
    )

    theta0 = 0.55
    crank_pin = crank_radius * np.array((np.cos(theta0), np.sin(theta0)))
    lever_angle = np.arctan2(crank_pin[1], crank_pin[0] - pivot_distance)
    lever_tip = np.array(
        (
            pivot_distance - lever_arm * np.cos(lever_angle),
            -lever_arm * np.sin(lever_angle),
        )
    )
    horizontal_offset = pivot_distance - lever_tip[0]
    vertical_reach = np.sqrt(rod_length**2 - horizontal_offset**2)
    ram_y = lever_tip[1] + vertical_reach
    rod_angle = np.arctan2(
        ram_y - lever_tip[1],
        pivot_distance - lever_tip[0],
    )

    guess = {
        crank: (0.0, 0.0, theta0),
        block: (crank_pin[0], crank_pin[1], lever_angle),
        lever: (pivot_distance, 0.0, lever_angle),
        rod: (lever_tip[0], lever_tip[1], rod_angle),
        ram: (pivot_distance, ram_y, 0.0),
    }

    return (
        mechanism,
        input_joint,
        slot_joint,
        ram_joint,
        guess,
        theta0,
        crank_radius,
        pivot_distance,
        lever_arm,
        rod_length,
    )


def _circle_intersections(center_a, radius_a, center_b, radius_b):
    delta = center_b - center_a
    distance = np.linalg.norm(delta)
    along = (radius_a**2 - radius_b**2 + distance**2) / (2.0 * distance)
    height = np.sqrt(max(0.0, radius_a**2 - along**2))
    direction = delta / distance
    normal = np.array((-direction[1], direction[0]))
    base = center_a + along * direction
    return base + height * normal, base - height * normal


def _watt_ii():
    crank_radius = 0.07
    ground_o2_o4 = 0.28
    input_rod_length = 0.18
    ternary_b_local = np.array((0.20, 0.0))
    ternary_c_local = np.array((0.12, -0.10))
    tracer_local = np.array((0.08, 0.06))
    ground_o6 = np.array((0.325, 0.225))
    second_rod_length = 0.12
    output_rocker_length = 0.08

    mechanism = Mechanism("watt_ii_acceptance")
    ground = mechanism.ground
    ground_o2 = ground.add_point("O2", (0.0, 0.0))
    ground_o4 = ground.add_point("O4", (ground_o2_o4, 0.0))
    ground_o6_point = ground.add_point("O6", ground_o6)

    crank = mechanism.add_link("crank")
    crank_o2 = crank.add_point("O2", (0.0, 0.0))
    crank_a = crank.add_point("A", (crank_radius, 0.0))

    input_rod = mechanism.add_link("input_rod")
    rod_a = input_rod.add_point("A", (0.0, 0.0))
    rod_b = input_rod.add_point("B", (input_rod_length, 0.0))

    ternary = mechanism.add_link("ternary_rocker")
    ternary_o4 = ternary.add_point("O4", (0.0, 0.0))
    ternary_b = ternary.add_point("B", ternary_b_local)
    ternary_c = ternary.add_point("C", ternary_c_local)
    tracer = ternary.add_point("P", tracer_local)

    second_rod = mechanism.add_link("second_rod")
    second_c = second_rod.add_point("C", (0.0, 0.0))
    second_d = second_rod.add_point("D", (second_rod_length, 0.0))

    output_rocker = mechanism.add_link("output_rocker")
    output_d = output_rocker.add_point("D", (0.0, 0.0))
    output_o6 = output_rocker.add_point("O6", (output_rocker_length, 0.0))

    input_joint = mechanism.revolute(ground_o2, crank_o2, name="crank_input")
    mechanism.revolute(crank_a, rod_a)
    mechanism.revolute(rod_b, ternary_b)
    mechanism.revolute(ternary_o4, ground_o4)
    mechanism.revolute(ternary_c, second_c)
    mechanism.revolute(second_d, output_d)
    mechanism.revolute(output_o6, ground_o6_point)

    theta0 = 0.60
    o4 = np.array((ground_o2_o4, 0.0))
    point_a = crank_radius * np.array((np.cos(theta0), np.sin(theta0)))
    candidates_b = _circle_intersections(
        point_a,
        input_rod_length,
        o4,
        np.linalg.norm(ternary_b_local),
    )
    point_b = max(candidates_b, key=lambda point: point[1])

    ternary_angle = np.arctan2(point_b[1], point_b[0] - ground_o2_o4)
    rotation = np.array(
        (
            (np.cos(ternary_angle), -np.sin(ternary_angle)),
            (np.sin(ternary_angle), np.cos(ternary_angle)),
        )
    )
    point_c = o4 + rotation @ ternary_c_local
    candidates_d = _circle_intersections(
        point_c,
        second_rod_length,
        ground_o6,
        output_rocker_length,
    )
    point_d = min(candidates_d, key=lambda point: point[1])

    input_rod_angle = np.arctan2(
        point_b[1] - point_a[1],
        point_b[0] - point_a[0],
    )
    second_rod_angle = np.arctan2(
        point_d[1] - point_c[1],
        point_d[0] - point_c[0],
    )
    output_angle = np.arctan2(
        ground_o6[1] - point_d[1],
        ground_o6[0] - point_d[0],
    )

    guess = {
        crank: (0.0, 0.0, theta0),
        input_rod: (point_a[0], point_a[1], input_rod_angle),
        ternary: (ground_o2_o4, 0.0, ternary_angle),
        second_rod: (point_c[0], point_c[1], second_rod_angle),
        output_rocker: (point_d[0], point_d[1], output_angle),
    }

    constants = {
        "crank_radius": crank_radius,
        "ground_o2_o4": ground_o2_o4,
        "input_rod_length": input_rod_length,
        "ternary_b_local": ternary_b_local,
        "ternary_c_local": ternary_c_local,
        "tracer_local": tracer_local,
        "ground_o6": ground_o6,
        "second_rod_length": second_rod_length,
        "output_rocker_length": output_rocker_length,
    }
    return (
        mechanism,
        input_joint,
        ternary_b,
        ternary_c,
        output_d,
        tracer,
        guess,
        theta0,
        constants,
    )


def _watt_analytical(values, constants):
    o4 = np.array((constants["ground_o2_o4"], 0.0))
    previous_b = None
    previous_d = None
    points_b = []
    points_c = []
    points_d = []
    tracers = []

    for theta in values:
        point_a = constants["crank_radius"] * np.array(
            (np.cos(theta), np.sin(theta))
        )
        candidates_b = _circle_intersections(
            point_a,
            constants["input_rod_length"],
            o4,
            np.linalg.norm(constants["ternary_b_local"]),
        )
        if previous_b is None:
            point_b = max(candidates_b, key=lambda point: point[1])
        else:
            point_b = min(
                candidates_b,
                key=lambda point: np.linalg.norm(point - previous_b),
            )
        previous_b = point_b

        angle = np.arctan2(
            point_b[1] - o4[1],
            point_b[0] - o4[0],
        )
        rotation = np.array(
            (
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            )
        )
        point_c = o4 + rotation @ constants["ternary_c_local"]
        tracer = o4 + rotation @ constants["tracer_local"]

        candidates_d = _circle_intersections(
            point_c,
            constants["second_rod_length"],
            constants["ground_o6"],
            constants["output_rocker_length"],
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
        np.asarray(points_b),
        np.asarray(points_c),
        np.asarray(points_d),
        np.asarray(tracers),
    )


def test_archimedes_trammel_full_cycle_matches_analytical_geometry():
    (
        mechanism,
        input_joint,
        horizontal_joint,
        vertical_joint,
        tracer,
        guess,
        theta0,
        bar_length,
        tracer_distance,
    ) = _archimedes_trammel()

    values = np.linspace(theta0, theta0 + 2.0 * np.pi, 73)
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values,
        ),
        initial_guess=guess,
    )

    expected_path = np.column_stack(
        (
            -(bar_length - tracer_distance) * np.cos(values),
            tracer_distance * np.sin(values),
        )
    )
    np.testing.assert_allclose(
        solution.point_positions(tracer),
        expected_path,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        solution.joint_coordinates(horizontal_joint),
        -bar_length * np.cos(values),
        atol=2e-9,
    )
    np.testing.assert_allclose(
        solution.joint_coordinates(vertical_joint),
        bar_length * np.sin(values),
        atol=2e-9,
    )
    assert np.all(solution.diagnostics.ranks == 9)
    assert np.all(solution.diagnostics.residual_norms <= 1e-9)


def test_whitworth_full_cycle_matches_slot_and_ram_geometry():
    (
        mechanism,
        input_joint,
        slot_joint,
        ram_joint,
        guess,
        theta0,
        crank_radius,
        pivot_distance,
        lever_arm,
        rod_length,
    ) = _whitworth()

    values = np.linspace(theta0, theta0 + 2.0 * np.pi, 73)
    solution = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values,
        ),
        initial_guess=guess,
    )

    crank_x = crank_radius * np.cos(values)
    crank_y = crank_radius * np.sin(values)
    expected_slot = np.sqrt(
        pivot_distance**2
        + crank_radius**2
        - 2.0 * pivot_distance * crank_radius * np.cos(values)
    )
    lever_angle = np.arctan2(crank_y, crank_x - pivot_distance)
    lever_tip_x = pivot_distance - lever_arm * np.cos(lever_angle)
    lever_tip_y = -lever_arm * np.sin(lever_angle)
    horizontal_offset = pivot_distance - lever_tip_x
    expected_ram = (
        lever_tip_y
        + np.sqrt(rod_length**2 - horizontal_offset**2)
    )

    np.testing.assert_allclose(
        solution.joint_coordinates(slot_joint),
        expected_slot,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        solution.joint_coordinates(ram_joint),
        expected_ram,
        atol=3e-9,
    )
    assert np.all(solution.diagnostics.ranks == 15)
    assert np.all(solution.diagnostics.residual_norms <= 1e-9)


def test_watt_ii_full_cycle_matches_independent_geometry_and_reverse_branch():
    (
        mechanism,
        input_joint,
        point_b,
        point_c,
        point_d,
        tracer,
        guess,
        theta0,
        constants,
    ) = _watt_ii()

    values = np.linspace(theta0, theta0 + 2.0 * np.pi, 73)
    forward = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values,
        ),
        initial_guess=guess,
    )
    reverse = solve(
        mechanism,
        driver=KinematicDriver(
            input_joint,
            position=values[::-1],
        ),
        initial_guess=forward[-1],
    )[::-1]

    expected_b, expected_c, expected_d, expected_tracer = _watt_analytical(
        values,
        constants,
    )
    np.testing.assert_allclose(
        forward.point_positions(point_b),
        expected_b,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        forward.point_positions(point_c),
        expected_c,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        forward.point_positions(point_d),
        expected_d,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        forward.point_positions(tracer),
        expected_tracer,
        atol=5e-9,
    )

    for link in mechanism.links:
        forward_poses = forward.body_poses(link)
        reverse_poses = reverse.body_poses(link)
        np.testing.assert_allclose(
            forward_poses[:, :2],
            reverse_poses[:, :2],
            atol=2e-9,
        )
        assert np.max(
            np.abs(
                _wrapped_angle_difference(
                    forward_poses[:, 2],
                    reverse_poses[:, 2],
                )
            )
        ) < 2e-9

    assert np.all(forward.diagnostics.ranks == 15)
    assert np.all(reverse.diagnostics.ranks == 15)
    assert np.all(forward.diagnostics.residual_norms <= 1e-9)
    assert np.all(reverse.diagnostics.residual_norms <= 1e-9)
