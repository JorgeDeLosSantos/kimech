"""Render structural mechanism topology independently of solved geometry."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from ..joints import PrismaticJoint, RevoluteJoint
from ..model import Ground, Link, Mechanism
from ..topology import MechanismTopology

_Body = Ground | Link
_Joint = RevoluteJoint | PrismaticJoint


def plot_topology(mechanism_or_topology: Mechanism | MechanismTopology, *, ax=None):
    """Plot structural body/joint connectivity and return figure and axes.

    The layout is schematic and intentionally independent of mechanism geometry
    and solved configuration coordinates.
    """
    topology = _as_topology(mechanism_or_topology)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    positions = _topology_layout(topology)
    parallel_groups = _parallel_joint_groups(topology)

    for joint_index, joint in enumerate(topology.joints):
        body_a = joint.point_a.body
        body_b = joint.point_b.body
        group = parallel_groups[_body_pair_key(topology, body_a, body_b)]
        offset_index = group.index(joint)
        curvature = _parallel_curvature(offset_index, len(group))
        center = _draw_joint_edge(
            positions[body_a],
            positions[body_b],
            joint,
            joint_index,
            curvature,
            ax,
        )
        _draw_joint_marker(center, joint, joint_index, ax)

    for body in topology.bodies:
        _draw_body_node(body, positions[body], ax)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.25)
    ax.set_axis_off()
    return fig, ax


def _as_topology(value: Mechanism | MechanismTopology) -> MechanismTopology:
    if isinstance(value, MechanismTopology):
        return value
    if isinstance(value, Mechanism):
        return value.topology()
    raise TypeError("mechanism_or_topology must be a Mechanism or MechanismTopology")


def _topology_layout(topology: MechanismTopology) -> dict[_Body, np.ndarray]:
    """Return deterministic display coordinates for topology bodies."""
    positions: dict[_Body, np.ndarray] = {}
    component_count = len(topology.connected_components)

    for component_index, component in enumerate(topology.connected_components):
        center_x = 3.5 * (component_index - 0.5 * (component_count - 1))
        center = np.asarray([center_x, 0.0], dtype=float)

        if len(component) == 1:
            positions[component[0]] = center
            continue

        radius = 1.0 + 0.12 * max(0, len(component) - 2)
        for index, body in enumerate(component):
            angle = 0.5 * np.pi - 2.0 * np.pi * index / len(component)
            positions[body] = center + radius * np.asarray(
                [np.cos(angle), np.sin(angle)],
                dtype=float,
            )

    return positions


def _parallel_joint_groups(
    topology: MechanismTopology,
) -> dict[tuple[int, int], list[_Joint]]:
    groups: dict[tuple[int, int], list[_Joint]] = defaultdict(list)
    for joint in topology.joints:
        groups[_body_pair_key(topology, joint.point_a.body, joint.point_b.body)].append(
            joint
        )
    return groups


def _body_pair_key(
    topology: MechanismTopology,
    body_a: _Body,
    body_b: _Body,
) -> tuple[int, int]:
    body_indices = {body: index for index, body in enumerate(topology.bodies)}
    index_a = body_indices[body_a]
    index_b = body_indices[body_b]
    return (min(index_a, index_b), max(index_a, index_b))


def _parallel_curvature(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    spacing = 0.24
    return spacing * (index - 0.5 * (count - 1))


def _draw_joint_edge(
    start: np.ndarray,
    end: np.ndarray,
    joint: _Joint,
    joint_index: int,
    curvature: float,
    ax,
) -> np.ndarray:
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance <= np.finfo(float).eps:
        perpendicular = np.asarray([0.0, 0.0])
    else:
        perpendicular = np.asarray([-delta[1], delta[0]]) / distance

    midpoint = 0.5 * (start + end)
    control = midpoint + curvature * distance * perpendicular

    path = Path(
        [start, control, end],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3],
    )
    linestyle = "--" if isinstance(joint, PrismaticJoint) else "-"
    edge = PathPatch(
        path,
        fill=False,
        edgecolor="0.35",
        linewidth=1.6,
        linestyle=linestyle,
        zorder=1,
    )
    edge.set_gid(f"kimech-topology-edge:{joint_index}")
    ax.add_patch(edge)

    return 0.25 * start + 0.5 * control + 0.25 * end


def _draw_joint_marker(
    center: np.ndarray,
    joint: _Joint,
    joint_index: int,
    ax,
):
    if isinstance(joint, RevoluteJoint):
        marker = "o"
        kind = "revolute"
    else:
        marker = "s"
        kind = "prismatic"

    artist = ax.scatter(
        [center[0]],
        [center[1]],
        marker=marker,
        s=58,
        facecolor="white",
        edgecolor="0.15",
        linewidth=1.4,
        zorder=3,
    )
    artist.set_gid(f"kimech-topology-joint:{kind}:{joint_index}")
    return artist


def _draw_body_node(body: _Body, position: np.ndarray, ax):
    is_ground = isinstance(body, Ground)
    marker = "s" if is_ground else "o"
    facecolor = "0.85" if is_ground else "white"

    artist = ax.scatter(
        [position[0]],
        [position[1]],
        marker=marker,
        s=520,
        facecolor=facecolor,
        edgecolor="0.15",
        linewidth=1.8,
        zorder=4,
    )
    kind = "ground" if is_ground else "link"
    artist.set_gid(f"kimech-topology-body:{kind}:{body.name}")

    label = ax.text(
        position[0],
        position[1],
        body.name,
        ha="center",
        va="center",
        fontsize=9,
        zorder=5,
    )
    label.set_gid(f"kimech-topology-label:{kind}:{body.name}")
    return artist, label
