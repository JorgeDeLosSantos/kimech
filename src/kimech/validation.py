"""Structural mobility and consistency validation."""

from __future__ import annotations

from dataclasses import dataclass

from .model import Ground, Link, Mechanism


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Lightweight structural validation result."""

    mobility: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether no structural errors were found."""
        return not self.errors


def structural_mobility(mechanism: Mechanism) -> int:
    """Return the planar lower-pair Grübler-Kutzbach mobility estimate.

    For the current R/P-only model every joint is a one-DOF lower pair, so

        M = 3 * n_mobile - 2 * n_joints.

    The result is a structural estimate and does not account for redundant
    constraints or special mechanism geometry.
    """
    if not isinstance(mechanism, Mechanism):
        raise TypeError("mechanism must be a Mechanism")
    return 3 * len(mechanism.links) - 2 * len(mechanism.joints)


def validate_mechanism(mechanism: Mechanism) -> ValidationReport:
    """Validate structural consistency without attempting numerical solution."""
    if not isinstance(mechanism, Mechanism):
        raise TypeError("mechanism must be a Mechanism")

    errors: list[str] = []
    warnings: list[str] = []
    mobility = structural_mobility(mechanism)

    if not mechanism.links:
        warnings.append("mechanism has no mobile links")

    known_bodies: set[Link | Ground] = {mechanism.ground, *mechanism.links}
    adjacency: dict[Link | Ground, set[Link | Ground]] = {
        body: set() for body in known_bodies
    }

    for index, joint in enumerate(mechanism.joints):
        body_a = joint.point_a.body
        body_b = joint.point_b.body

        if body_a not in known_bodies or body_b not in known_bodies:
            errors.append(
                f"joint at index {index} references a body not owned by this mechanism"
            )
            continue

        if body_a is body_b:
            errors.append(f"joint at index {index} connects a body to itself")
            continue

        adjacency[body_a].add(body_b)
        adjacency[body_b].add(body_a)

    reachable = _reachable_from_ground(mechanism.ground, adjacency)
    disconnected = [link.name for link in mechanism.links if link not in reachable]
    if disconnected:
        names = ", ".join(repr(name) for name in disconnected)
        errors.append(f"mobile links disconnected from ground: {names}")

    if mobility < 0:
        warnings.append(
            "structural mobility estimate is negative; redundant constraints "
            "or an overconstrained topology may be present"
        )

    return ValidationReport(
        mobility=mobility,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _reachable_from_ground(
    ground: Ground,
    adjacency: dict[Link | Ground, set[Link | Ground]],
) -> set[Link | Ground]:
    reachable: set[Link | Ground] = set()
    pending: list[Link | Ground] = [ground]

    while pending:
        body = pending.pop()
        if body in reachable:
            continue
        reachable.add(body)
        pending.extend(adjacency.get(body, ()))

    return reachable
