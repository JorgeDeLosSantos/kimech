import numpy as np
import pytest

from kimech.driver import KinematicDriver
from kimech.joints import RevoluteJoint
from kimech import Mechanism


def _revolute():
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("O", (0.0, 0.0))
    link = mechanism.add_link("link")
    moving = link.add_point("O", (0.0, 0.0))
    return mechanism.revolute(fixed, moving)


def test_driver_accepts_scalar_position_and_differential_data():
    joint = _revolute()
    driver = KinematicDriver(
        joint,
        position=0.5,
        velocity=2.0,
        acceleration=-0.3,
    )

    assert driver.joint is joint
    assert driver.position == pytest.approx(0.5)
    assert driver.velocity == pytest.approx(2.0)
    assert driver.acceleration == pytest.approx(-0.3)
    assert driver.sample_count == 1
    assert driver.is_scalar


def test_driver_broadcasts_scalar_differential_data_over_position_history():
    joint = _revolute()
    position = np.array([0.2, 0.4, 0.6])
    driver = KinematicDriver(
        joint,
        position=position,
        velocity=3.0,
        acceleration=0.0,
    )

    np.testing.assert_array_equal(driver.position, position)
    assert driver.velocity == pytest.approx(3.0)
    assert driver.acceleration == pytest.approx(0.0)
    np.testing.assert_array_equal(driver._velocity_history(), [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(driver._acceleration_history(), [0.0, 0.0, 0.0])
    assert len(driver) == 3
    assert not driver.is_scalar


def test_driver_accepts_elementwise_differential_histories():
    joint = _revolute()
    driver = KinematicDriver(
        joint,
        position=[0.2, 0.4, 0.6],
        velocity=[1.0, -2.0, 0.0],
        acceleration=[0.1, 0.2, 0.3],
    )

    np.testing.assert_array_equal(driver.velocity, [1.0, -2.0, 0.0])
    np.testing.assert_array_equal(driver.acceleration, [0.1, 0.2, 0.3])


def test_driver_public_arrays_are_safe_copies():
    joint = _revolute()
    source = np.array([0.2, 0.4])
    driver = KinematicDriver(joint, position=source)

    source[0] = 99.0
    returned = driver.position
    returned[1] = 88.0

    np.testing.assert_array_equal(driver.position, [0.2, 0.4])


@pytest.mark.parametrize(
    "position, velocity, acceleration, message",
    [
        ([], None, None, "empty"),
        ([[0.2], [0.4]], None, None, "1-dimensional"),
        ([0.2, np.nan], None, None, "finite"),
        ("bad", None, None, "numeric"),
        (0.5, [1.0], None, "scalar"),
        ([0.2, 0.4], [1.0], None, "shape"),
        ([0.2, 0.4], [[1.0], [2.0]], None, "1-dimensional"),
        ([0.2, 0.4], [1.0, np.nan], None, "finite"),
        ([0.2, 0.4], None, [0.0, 0.0], "requires velocity"),
    ],
)
def test_driver_validates_prescribed_data(position, velocity, acceleration, message):
    joint = _revolute()
    with pytest.raises((TypeError, ValueError), match=message):
        KinematicDriver(
            joint,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )


def test_driver_rejects_unsupported_target_and_attribute_mutation():
    with pytest.raises(TypeError, match="RevoluteJoint or PrismaticJoint"):
        KinematicDriver(object(), position=0.0)

    driver = KinematicDriver(_revolute(), position=0.0)
    with pytest.raises(AttributeError, match="immutable"):
        driver.position = 1.0

    assert isinstance(driver.joint, RevoluteJoint)
