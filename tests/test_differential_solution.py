import numpy as np
import pytest

from kimech import KinematicSolution, Mechanism


def _mechanism():
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("fixed", (0.0, 0.0))
    link = mechanism.add_link("link")
    moving = link.add_point("moving", (0.0, 0.0))
    point = link.add_point("P", (1.0, 0.0))
    joint = mechanism.revolute(fixed, moving)
    return mechanism, link, point, joint


def test_solution_stores_differential_histories_and_preserves_them_when_indexed():
    mechanism, link, _, joint = _mechanism()
    values = np.array([0.0, 0.5])
    coordinates = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.5]])
    velocities = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    accelerations = np.array([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]])
    input_velocities = np.array([15.0, 16.0])
    input_accelerations = np.array([17.0, 18.0])

    solution = KinematicSolution(
        mechanism,
        joint,
        values,
        coordinates,
        coordinate_velocities=velocities,
        coordinate_accelerations=accelerations,
        input_velocities=input_velocities,
        input_accelerations=input_accelerations,
    )

    assert solution.has_velocity is True
    assert solution.has_acceleration is True
    np.testing.assert_allclose(solution.coordinate_velocities, velocities)
    np.testing.assert_allclose(solution.coordinate_accelerations, accelerations)
    np.testing.assert_allclose(solution.input_velocities, input_velocities)
    np.testing.assert_allclose(solution.input_accelerations, input_accelerations)
    np.testing.assert_allclose(solution.body_velocities(link), velocities)
    np.testing.assert_allclose(solution.body_accelerations(link), accelerations)

    config = solution[1]
    assert config.has_velocity is True
    assert config.has_acceleration is True
    assert config.input_velocity == pytest.approx(16.0)
    assert config.input_acceleration == pytest.approx(18.0)
    np.testing.assert_allclose(config.coordinate_velocities, velocities[1])
    np.testing.assert_allclose(config.coordinate_accelerations, accelerations[1])


def test_solution_differential_entity_histories_have_expected_values():
    mechanism, link, point, joint = _mechanism()
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocities = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    accelerations = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    solution = KinematicSolution(
        mechanism,
        joint,
        [0.0, 0.0],
        coordinates,
        coordinate_velocities=velocities,
        coordinate_accelerations=accelerations,
    )

    expected_point_velocities = np.array([[1.0, 5.0], [4.0, 11.0]])
    expected_point_accelerations = np.array([[-2.0, 17.0], [-26.0, 23.0]])

    np.testing.assert_allclose(solution.point_velocities(point), expected_point_velocities)
    np.testing.assert_allclose(
        solution.point_accelerations(point),
        expected_point_accelerations,
    )
    np.testing.assert_allclose(solution.joint_velocities(joint), [3.0, 6.0])
    np.testing.assert_allclose(solution.joint_accelerations(joint), [9.0, 12.0])
    np.testing.assert_allclose(
        solution.body_velocities(mechanism.ground),
        np.zeros((2, 3)),
    )
    np.testing.assert_allclose(
        solution.body_accelerations(mechanism.ground),
        np.zeros((2, 3)),
    )


def test_solution_unavailable_differential_state_fails_explicitly():
    mechanism, link, point, joint = _mechanism()
    solution = KinematicSolution(mechanism, joint, [0.0], [[0.0, 0.0, 0.0]])

    assert solution.has_velocity is False
    assert solution.has_acceleration is False
    assert solution.input_velocities is None
    assert solution.input_accelerations is None

    with pytest.raises(ValueError, match="velocity data"):
        _ = solution.coordinate_velocities
    with pytest.raises(ValueError, match="acceleration data"):
        _ = solution.coordinate_accelerations
    with pytest.raises(ValueError, match="velocity data"):
        solution.body_velocities(link)
    with pytest.raises(ValueError, match="velocity data"):
        solution.point_velocities(point)
    with pytest.raises(ValueError, match="velocity data"):
        solution.joint_velocities(joint)
    with pytest.raises(ValueError, match="acceleration data"):
        solution.body_accelerations(link)
    with pytest.raises(ValueError, match="acceleration data"):
        solution.point_accelerations(point)
    with pytest.raises(ValueError, match="acceleration data"):
        solution.joint_accelerations(joint)


def test_solution_differential_arrays_are_safe_copies():
    mechanism, _, _, joint = _mechanism()
    velocities = np.array([[1.0, 2.0, 3.0]])
    accelerations = np.array([[4.0, 5.0, 6.0]])
    input_velocities = np.array([7.0])
    input_accelerations = np.array([8.0])
    solution = KinematicSolution(
        mechanism,
        joint,
        [0.0],
        [[0.0, 0.0, 0.0]],
        coordinate_velocities=velocities,
        coordinate_accelerations=accelerations,
        input_velocities=input_velocities,
        input_accelerations=input_accelerations,
    )

    velocities[:] = -1.0
    accelerations[:] = -1.0
    input_velocities[:] = -1.0
    input_accelerations[:] = -1.0

    exposed_velocity = solution.coordinate_velocities
    exposed_acceleration = solution.coordinate_accelerations
    exposed_input_velocity = solution.input_velocities
    exposed_input_acceleration = solution.input_accelerations
    exposed_velocity[:] = 99.0
    exposed_acceleration[:] = 99.0
    exposed_input_velocity[:] = 99.0
    exposed_input_acceleration[:] = 99.0

    np.testing.assert_allclose(solution.coordinate_velocities, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(solution.coordinate_accelerations, [[4.0, 5.0, 6.0]])
    np.testing.assert_allclose(solution.input_velocities, [7.0])
    np.testing.assert_allclose(solution.input_accelerations, [8.0])


def test_solution_validates_differential_shapes_and_invariants():
    mechanism, _, _, joint = _mechanism()
    coordinates = [[0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="requires coordinate_velocities"):
        KinematicSolution(
            mechanism,
            joint,
            [0.0],
            coordinates,
            coordinate_accelerations=[[0.0, 0.0, 0.0]],
        )
    with pytest.raises(ValueError, match="requires input_velocities"):
        KinematicSolution(
            mechanism,
            joint,
            [0.0],
            coordinates,
            input_accelerations=[0.0],
        )
    with pytest.raises(ValueError, match="shape"):
        KinematicSolution(
            mechanism,
            joint,
            [0.0],
            coordinates,
            coordinate_velocities=[[0.0, 0.0]],
        )
    with pytest.raises(ValueError, match="shape"):
        KinematicSolution(
            mechanism,
            joint,
            [0.0],
            coordinates,
            input_velocities=[0.0, 1.0],
        )
    with pytest.raises(ValueError, match="finite"):
        KinematicSolution(
            mechanism,
            joint,
            [0.0],
            coordinates,
            coordinate_velocities=[[0.0, np.inf, 0.0]],
        )


def test_empty_differential_solution_preserves_history_shapes():
    mechanism, link, point, joint = _mechanism()
    solution = KinematicSolution(
        mechanism,
        joint,
        np.empty(0),
        np.empty((0, 3)),
        coordinate_velocities=np.empty((0, 3)),
        coordinate_accelerations=np.empty((0, 3)),
        input_velocities=np.empty(0),
        input_accelerations=np.empty(0),
    )

    assert solution.coordinate_velocities.shape == (0, 3)
    assert solution.coordinate_accelerations.shape == (0, 3)
    assert solution.input_velocities.shape == (0,)
    assert solution.input_accelerations.shape == (0,)
    assert solution.body_velocities(link).shape == (0, 3)
    assert solution.body_accelerations(link).shape == (0, 3)
    assert solution.point_velocities(point).shape == (0, 2)
    assert solution.point_accelerations(point).shape == (0, 2)
    assert solution.joint_velocities(joint).shape == (0,)
    assert solution.joint_accelerations(joint).shape == (0,)
