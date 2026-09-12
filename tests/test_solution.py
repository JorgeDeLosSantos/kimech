import numpy as np
import pytest

from kimech import Configuration, KinematicSolution, Mechanism


def _revolute_mechanism():
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("fixed", (0.0, 0.0))
    link = mechanism.add_link("link")
    moving = link.add_point("moving", (0.0, 0.0))
    joint = mechanism.revolute(fixed, moving)
    return mechanism, link, joint


def test_configuration_body_poses_follow_link_creation_order_and_ground_is_fixed():
    mechanism = Mechanism()
    first = mechanism.add_link("first")
    second = mechanism.add_link("second")
    config = Configuration(mechanism, [1.0, 2.0, 0.5, 4.0, 5.0, 0.75])

    np.testing.assert_allclose(config.body_pose(first), [1.0, 2.0, 0.5])
    np.testing.assert_allclose(config.body_pose(second), [4.0, 5.0, 0.75])
    np.testing.assert_allclose(config.body_pose(mechanism.ground), [0.0, 0.0, 0.0])


def test_position_transforms_link_point_from_arbitrary_local_frame():
    mechanism = Mechanism()
    link = mechanism.add_link("link")
    point = link.add_point("P", (1.0, 0.0))
    config = Configuration(mechanism, [2.0, 3.0, np.pi / 2])

    np.testing.assert_allclose(config.position(point), [2.0, 4.0], atol=1e-12)

    offset = link.add_point("offset", (2.0, -1.0))
    expected = [2.0 - np.sqrt(2) / 2, 3.0 + 3 * np.sqrt(2) / 2]
    rotated_config = Configuration(mechanism, [2.0, 3.0, 3 * np.pi / 4])
    np.testing.assert_allclose(rotated_config.position(offset), expected)


def test_ground_point_position_is_independent_of_mobile_coordinates():
    mechanism = Mechanism()
    point = mechanism.ground.add_point("A", (3.0, 4.0))
    mechanism.add_link("link")

    np.testing.assert_allclose(Configuration(mechanism, [9.0, -2.0, 8.0]).position(point), [3.0, 4.0])


def test_configuration_keeps_link_layout_after_mechanism_gains_a_link():
    mechanism = Mechanism()
    original = mechanism.add_link("a")
    config = Configuration(mechanism, [1.0, 2.0, 0.3])

    added = mechanism.add_link("b")

    np.testing.assert_allclose(config.body_pose(original), [1.0, 2.0, 0.3])
    with pytest.raises(ValueError):
        config.body_pose(added)


def test_configuration_accepts_new_point_only_on_a_snapshotted_link():
    mechanism = Mechanism()
    original = mechanism.add_link("a")
    config = Configuration(mechanism, [1.0, 2.0, 0.0])

    original_point = original.add_point("P", (1.0, 0.0))
    added = mechanism.add_link("b")
    added_point = added.add_point("P", (0.0, 0.0))

    np.testing.assert_allclose(config.position(original_point), [2.0, 2.0])
    with pytest.raises(ValueError):
        config.position(added_point)


def test_configuration_rejects_new_joint_using_link_outside_snapshot():
    mechanism = Mechanism()
    original = mechanism.add_link("a")
    original_point = original.add_point("P", (0.0, 0.0))
    config = Configuration(mechanism, [1.0, 2.0, 0.0])

    added = mechanism.add_link("b")
    added_point = added.add_point("P", (0.0, 0.0))
    new_joint = mechanism.revolute(original_point, added_point)

    with pytest.raises(ValueError):
        config.joint_coordinate(new_joint)


def test_revolute_coordinate_respects_a_to_b_order_and_is_unwrapped():
    mechanism = Mechanism()
    ground_point = mechanism.ground.add_point("G", (0.0, 0.0))
    first = mechanism.add_link("first")
    first_point = first.add_point("P", (0.0, 0.0))
    second = mechanism.add_link("second")
    second_point = second.add_point("P", (0.0, 0.0))
    ground_to_first = mechanism.revolute(ground_point, first_point)
    first_to_second = mechanism.revolute(first_point, second_point)
    config = Configuration(mechanism, [0.0, 0.0, 0.25, 0.0, 0.0, 0.25 + 3 * np.pi])

    assert config.joint_coordinate(ground_to_first) == pytest.approx(0.25)
    assert config.joint_coordinate(first_to_second) == pytest.approx(3 * np.pi)


def test_horizontal_prismatic_coordinate_projects_relative_reference_points():
    mechanism = Mechanism()
    point_a = mechanism.ground.add_point("A", (1.0, 2.0))
    slider = mechanism.add_link("slider")
    point_b = slider.add_point("B", (0.5, -1.0))
    joint = mechanism.prismatic(point_a, point_b, axis_a=(1.0, 0.0), axis_b=(1.0, 0.0))
    config = Configuration(mechanism, [5.0, 3.0, 0.0])

    assert config.joint_coordinate(joint) == pytest.approx(4.5)


def test_inclined_prismatic_axis_rotates_with_mobile_body_a():
    mechanism = Mechanism()
    body_a = mechanism.add_link("a")
    point_a = body_a.add_point("A", (1.0, 0.0))
    body_b = mechanism.add_link("b")
    point_b = body_b.add_point("B", (0.0, 0.0))
    alpha = np.pi / 6
    joint = mechanism.prismatic(
        point_a,
        point_b,
        axis_a=(np.cos(alpha), np.sin(alpha)),
        axis_b=(1.0, 0.0),
    )
    theta_a = np.pi / 3
    config = Configuration(mechanism, [2.0, 1.0, theta_a, 0.0, 5.0, 0.0])

    position_a = np.array([2.0, 1.0]) + np.array([np.cos(theta_a), np.sin(theta_a)])
    displacement = np.array([0.0, 5.0]) - position_a
    global_axis = [np.cos(alpha + theta_a), np.sin(alpha + theta_a)]
    assert config.joint_coordinate(joint) == pytest.approx(np.dot(global_axis, displacement))


def test_solution_sequence_properties_and_negative_indexing():
    mechanism, link, joint = _revolute_mechanism()
    inputs = np.array([0.0, 0.5, 1.0])
    coordinates = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.5], [3.0, 4.0, 1.0]])
    solution = KinematicSolution(mechanism, joint, inputs, coordinates)

    assert len(solution) == 3
    assert solution.mechanism is mechanism
    assert solution.input_joint is joint
    assert solution[0].input_joint is joint
    assert solution[0].input_value == pytest.approx(0.0)
    np.testing.assert_allclose(solution[0].body_pose(link), coordinates[0])
    np.testing.assert_allclose(solution[-1].body_pose(link), coordinates[-1])
    np.testing.assert_allclose(solution.input_values, inputs)
    np.testing.assert_allclose(solution.coordinates, coordinates)

    with pytest.raises(TypeError):
        solution[:2]


def test_solution_and_derived_configuration_keep_original_link_layout():
    mechanism, original, joint = _revolute_mechanism()
    solution = KinematicSolution(mechanism, joint, [0.5], [[1.0, 2.0, 0.5]])

    added = mechanism.add_link("added")
    added_point = added.add_point("P", (0.0, 0.0))
    added_joint = mechanism.revolute(original.points[0], added_point)
    config = solution[0]

    np.testing.assert_allclose(solution.body_poses(original), [[1.0, 2.0, 0.5]])
    np.testing.assert_allclose(config.body_pose(original), [1.0, 2.0, 0.5])
    with pytest.raises(ValueError):
        solution.body_poses(added)
    with pytest.raises(ValueError):
        solution.point_path(added_point)
    with pytest.raises(ValueError):
        solution.joint_coordinates(added_joint)
    with pytest.raises(ValueError):
        config.body_pose(added)


def test_point_path_body_poses_and_joint_coordinates_have_expected_values():
    mechanism, link, joint = _revolute_mechanism()
    point = link.add_point("P", (1.0, 0.0))
    angles = np.array([0.0, np.pi / 2, np.pi])
    coordinates = np.column_stack((np.full(3, 2.0), np.full(3, 3.0), angles))
    solution = KinematicSolution(mechanism, joint, angles, coordinates)

    expected_path = [[3.0, 3.0], [2.0, 4.0], [1.0, 3.0]]
    assert solution.point_path(point).shape == (3, 2)
    np.testing.assert_allclose(solution.point_path(point), expected_path, atol=1e-12)
    assert solution.body_poses(link).shape == (3, 3)
    np.testing.assert_allclose(solution.body_poses(link), coordinates)
    np.testing.assert_allclose(solution.body_poses(mechanism.ground), np.zeros((3, 3)))
    assert solution.joint_coordinates(joint).shape == (3,)
    np.testing.assert_allclose(solution.joint_coordinates(joint), angles)


def test_empty_solution_queries_preserve_documented_shapes():
    mechanism, link, joint = _revolute_mechanism()
    point = link.add_point("P", (1.0, 0.0))
    solution = KinematicSolution(mechanism, joint, np.empty(0), np.empty((0, 3)))

    assert solution.point_path(point).shape == (0, 2)
    assert solution.body_poses(link).shape == (0, 3)
    assert solution.body_poses(mechanism.ground).shape == (0, 3)
    assert solution.joint_coordinates(joint).shape == (0,)


def test_public_arrays_cannot_mutate_stored_results_or_alias_constructor_inputs():
    mechanism, link, joint = _revolute_mechanism()
    original_q = np.array([1.0, 2.0, 0.5])
    config = Configuration(mechanism, original_q)
    original_q[:] = -1.0
    exposed_q = config.coordinates
    exposed_q[:] = 99.0
    np.testing.assert_allclose(config.body_pose(link), [1.0, 2.0, 0.5])

    input_values = np.array([0.0, 0.5])
    coordinates = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.5]])
    solution = KinematicSolution(mechanism, joint, input_values, coordinates)
    input_values[:] = -1.0
    coordinates[:] = -1.0
    exposed_values = solution.input_values
    exposed_coordinates = solution.coordinates
    exposed_values[:] = 99.0
    exposed_coordinates[:] = 99.0

    np.testing.assert_allclose(solution.input_values, [0.0, 0.5])
    np.testing.assert_allclose(solution.coordinates, [[1.0, 2.0, 0.0], [2.0, 3.0, 0.5]])


def test_configuration_validates_arrays_input_metadata_and_membership():
    mechanism, _, joint = _revolute_mechanism()
    other, other_link, other_joint = _revolute_mechanism()

    with pytest.raises(TypeError, match="Mechanism"):
        Configuration(object(), [])
    with pytest.raises(TypeError, match="numeric"):
        Configuration(mechanism, [object(), 0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        Configuration(mechanism, [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        Configuration(mechanism, [0.0, np.inf, 0.0])
    with pytest.raises(ValueError, match="joint"):
        Configuration(mechanism, [0.0, 0.0, 0.0], input_joint=other_joint)
    with pytest.raises(ValueError, match="scalar"):
        Configuration(mechanism, [0.0, 0.0, 0.0], input_joint=joint, input_value=[1.0])
    with pytest.raises(ValueError, match="finite"):
        Configuration(mechanism, [0.0, 0.0, 0.0], input_value=np.nan)

    config = Configuration(mechanism, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="body"):
        config.body_pose(other_link)
    with pytest.raises(ValueError, match="point"):
        config.position(other_link.points[0])
    with pytest.raises(ValueError, match="joint"):
        config.joint_coordinate(other_joint)
    assert other is other_link.mechanism


def test_solution_validates_shapes_finiteness_and_external_entities():
    mechanism, _, joint = _revolute_mechanism()
    other, other_link, other_joint = _revolute_mechanism()
    solution = KinematicSolution(mechanism, joint, [0.0], [[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="joint"):
        KinematicSolution(mechanism, other_joint, [0.0], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="1-dimensional"):
        KinematicSolution(mechanism, joint, [[0.0]], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="2-dimensional"):
        KinematicSolution(mechanism, joint, [0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        KinematicSolution(mechanism, joint, [0.0, 1.0], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        KinematicSolution(mechanism, joint, [np.inf], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        KinematicSolution(mechanism, joint, [0.0], [[0.0, np.nan, 0.0]])

    external_point = other_link.add_point("external", (1.0, 0.0))
    with pytest.raises(ValueError):
        solution.point_path(external_point)
    with pytest.raises(ValueError):
        solution.body_poses(other_link)
    with pytest.raises(ValueError):
        solution.body_poses(other.ground)
    with pytest.raises(ValueError):
        solution.joint_coordinates(other_joint)
