"""Normalize sampled kinematic prescriptions for multi-DOF consumers.

The numerical solver will consume values sampled from this representation;
the normalization deliberately knows nothing about physical sample times.
"""

from __future__ import annotations

from collections.abc import Sequence

from .driver import KinematicDriver


def normalize_drivers(
    value: KinematicDriver | Sequence[KinematicDriver] | None,
    *,
    allow_empty: bool = False,
    single_sample: bool = False,
) -> tuple[KinematicDriver, ...]:
    """Return ordered, consistent drivers without changing their contents.

    The caller remains responsible for verifying that every selected joint
    belongs to the relevant mechanism/result snapshot.
    """
    if value is None:
        drivers: tuple[KinematicDriver, ...] = ()
    elif isinstance(value, KinematicDriver):
        drivers = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        drivers = tuple(value)
    else:
        raise TypeError("drivers must be a KinematicDriver or a sequence of KinematicDriver")

    if not drivers and not allow_empty:
        raise ValueError("at least one kinematic driver is required")

    seen_joints: set[int] = set()
    expected_count: int | None = None
    for index, driver in enumerate(drivers):
        if not isinstance(driver, KinematicDriver):
            raise TypeError(f"drivers[{index}] must be a KinematicDriver")
        joint_id = id(driver.joint)
        if joint_id in seen_joints:
            raise ValueError("multiple drivers cannot prescribe the same joint")
        seen_joints.add(joint_id)

        if expected_count is None:
            expected_count = driver.sample_count
        elif driver.sample_count != expected_count:
            raise ValueError(
                "all drivers must have the same number of position samples "
                f"({expected_count} != {driver.sample_count} at index {index})"
            )
        if single_sample and driver.sample_count != 1:
            raise ValueError("configuration drivers must contain exactly one sample")

    return drivers


def sampled_driver(
    driver: KinematicDriver,
    index: int | slice,
) -> KinematicDriver:
    """Copy one driver sample or a slice while retaining its joint identity."""
    positions = driver._position_history()
    velocities = driver._velocity_history()
    accelerations = driver._acceleration_history()

    if isinstance(index, slice):
        return KinematicDriver._from_history(
            driver.joint,
            positions[index],
            None if velocities is None else velocities[index],
            None if accelerations is None else accelerations[index],
        )

    return KinematicDriver(
        driver.joint,
        position=float(positions[index]),
        velocity=None if velocities is None else float(velocities[index]),
        acceleration=None if accelerations is None else float(accelerations[index]),
    )
