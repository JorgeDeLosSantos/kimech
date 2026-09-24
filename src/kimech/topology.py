"""Structural topology snapshots for planar mechanisms."""

from __future__ import annotations

from .joints import PrismaticJoint, RevoluteJoint
from .model import Ground, Link, Mechanism

_Body = Ground | Link
_Joint = RevoluteJoint | PrismaticJoint


class MechanismTopology:
    """Immutable-from-the-caller's-perspective structural mechanism snapshot.

    Topology is represented semantically as an undirected body-joint
    multigraph. Bodies are vertices and joints are edges, so multiple joints
    between the same pair of bodies retain their individual identities.
    """

    __slots__ = (
        "_adjacent",
        "_bodies",
        "_body_index",
        "_components",
        "_incident",
        "_joints",
        "_mechanism",
    )

    def __init__(self, mechanism: Mechanism) -> None:
        if not isinstance(mechanism, Mechanism):
            raise TypeError("mechanism must be a Mechanism")

        bodies: tuple[_Body, ...] = (mechanism.ground, *mechanism.links)
        joints: tuple[_Joint, ...] = mechanism.joints
        body_index = {body: index for index, body in enumerate(bodies)}

        incident_lists: dict[_Body, list[_Joint]] = {body: [] for body in bodies}
        adjacent_sets: dict[_Body, set[_Body]] = {body: set() for body in bodies}

        for joint in joints:
            body_a = joint.point_a.body
            body_b = joint.point_b.body
            if body_a not in body_index or body_b not in body_index:
                raise ValueError(
                    "mechanism contains a joint referencing a body outside "
                    "the topology snapshot"
                )
            if body_a is body_b:
                raise ValueError(
                    "mechanism contains a joint connecting a body to itself"
                )

            incident_lists[body_a].append(joint)
            incident_lists[body_b].append(joint)
            adjacent_sets[body_a].add(body_b)
            adjacent_sets[body_b].add(body_a)

        incident = {
            body: tuple(incident_lists[body])
            for body in bodies
        }
        adjacent = {
            body: tuple(
                candidate
                for candidate in bodies
                if candidate in adjacent_sets[body]
            )
            for body in bodies
        }

        self._mechanism = mechanism
        self._bodies = bodies
        self._joints = joints
        self._body_index = body_index
        self._incident = incident
        self._adjacent = adjacent
        self._components = _connected_components(bodies, adjacent)

    @property
    def mechanism(self) -> Mechanism:
        """Return the mechanism from which this topology snapshot was built."""
        return self._mechanism

    @property
    def bodies(self) -> tuple[_Body, ...]:
        """Return ground followed by mobile links in snapshot order."""
        return self._bodies

    @property
    def joints(self) -> tuple[_Joint, ...]:
        """Return joints in snapshot creation order."""
        return self._joints

    def incident_joints(self, body: _Body) -> tuple[_Joint, ...]:
        """Return joints incident to *body* in joint creation order."""
        validated = self._validate_body(body)
        return self._incident[validated]

    def adjacent_bodies(self, body: _Body) -> tuple[_Body, ...]:
        """Return unique bodies adjacent to *body* in topology body order."""
        validated = self._validate_body(body)
        return self._adjacent[validated]

    def joints_between(self, body_a: _Body, body_b: _Body) -> tuple[_Joint, ...]:
        """Return all joints connecting two distinct bodies."""
        validated_a = self._validate_body(body_a)
        validated_b = self._validate_body(body_b)
        if validated_a is validated_b:
            return ()
        return tuple(
            joint
            for joint in self._incident[validated_a]
            if _other_body(joint, validated_a) is validated_b
        )

    def degree(self, body: _Body) -> int:
        """Return multigraph degree, counting incident joint identities."""
        return len(self.incident_joints(body))

    @property
    def connected_components(self) -> tuple[tuple[_Body, ...], ...]:
        """Return deterministic body components for this snapshot."""
        return self._components

    @property
    def is_connected(self) -> bool:
        """Return whether all snapshot bodies belong to one component."""
        return len(self._components) == 1

    @property
    def cycle_rank(self) -> int:
        """Return the undirected multigraph cycle rank E - V + C."""
        return len(self._joints) - len(self._bodies) + len(self._components)

    def _validate_body(self, body: object) -> _Body:
        if not isinstance(body, (Ground, Link)):
            raise TypeError("body must be a Ground or Link")
        if body not in self._body_index:
            raise ValueError("body does not belong to this topology snapshot")
        return body


def _other_body(joint: _Joint, body: _Body) -> _Body:
    body_a = joint.point_a.body
    body_b = joint.point_b.body
    return body_b if body_a is body else body_a


def _connected_components(
    bodies: tuple[_Body, ...],
    adjacent: dict[_Body, tuple[_Body, ...]],
) -> tuple[tuple[_Body, ...], ...]:
    remaining = set(bodies)
    components: list[tuple[_Body, ...]] = []

    for start in bodies:
        if start not in remaining:
            continue

        reachable: set[_Body] = set()
        pending = [start]
        while pending:
            body = pending.pop()
            if body in reachable:
                continue
            reachable.add(body)
            pending.extend(
                neighbor
                for neighbor in reversed(adjacent[body])
                if neighbor not in reachable
            )

        component = tuple(body for body in bodies if body in reachable)
        components.append(component)
        remaining.difference_update(reachable)

    return tuple(components)
