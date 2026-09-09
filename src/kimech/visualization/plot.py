"""Render solved mechanism configurations as kinematic schematics."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from .._geometry import perpendicular, rotation_matrix
from ..joints import PrismaticJoint, RevoluteJoint
from ..model import Ground, Link, Point
from ..solution import Configuration

_Body = Link | Ground


def plot(config: Configuration, *, ax=None):
    """Plot one solved configuration and return its Matplotlib figure and axes."""
    if not isinstance(config, Configuration):
        raise TypeError("config must be a Configuration")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    mechanism = config.mechanism
    structural = _structural_points(config)
    scale = _plot_scale(config)
    body_colors: dict[_Body, str] = {mechanism.ground: "0.4"}

    _draw_body(config, mechanism.ground, structural[mechanism.ground], "0.4", ax)

    for link in mechanism.links:
        color = ax._get_lines.get_next_color()
        body_colors[link] = color
        _draw_body(config, link, structural[link], color, ax)

    for joint in mechanism.joints:
        if isinstance(joint, PrismaticJoint):
            _draw_prismatic_joint(config, joint, scale, ax)

    for body in (mechanism.ground, *mechanism.links):
        auxiliary = [point for point in body.points if id(point) not in structural[body][1]]
        if auxiliary:
            positions = np.asarray([config.position(point) for point in auxiliary])
            artist = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                s=22,
                color=body_colors[body],
                zorder=4,
            )
            artist.set_gid(f"kimech-auxiliary:{body.name}")

    for joint in mechanism.joints:
        if isinstance(joint, RevoluteJoint):
            _draw_revolute_joint(config, joint, ax)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def _structural_points(config: Configuration) -> dict[_Body, tuple[list[Point], set[int]]]:
    mechanism = config.mechanism
    result: dict[_Body, tuple[list[Point], set[int]]] = {
        body: ([], set()) for body in (mechanism.ground, *mechanism.links)
    }
    for joint in mechanism.joints:
        for point in (joint.point_a, joint.point_b):
            points, identities = result[point.body]
            identity = id(point)
            if identity not in identities:
                points.append(point)
                identities.add(identity)
    return result


def _plot_scale(config: Configuration) -> float:
    mechanism = config.mechanism
    positions = [
        config.position(point)
        for body in (mechanism.ground, *mechanism.links)
        for point in body.points
    ]
    if not positions:
        return 1.0
    coordinates = np.asarray(positions, dtype=float)
    extent = float(max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1])))
    return 1.0 if extent <= np.finfo(float).eps else extent


def _draw_body(
    config: Configuration,
    body: _Body,
    structural: tuple[list[Point], set[int]],
    color: str,
    ax,
) -> None:
    points, _ = structural
    if len(points) < 2:
        return

    positions = np.asarray([config.position(point) for point in points])
    if len(points) == 2:
        coordinates = positions
    else:
        hub = np.mean(positions, axis=0)
        coordinates = np.asarray(
            [coordinate for point in positions for coordinate in (hub, point, (np.nan, np.nan))]
        )

    is_ground = isinstance(body, Ground)
    (artist,) = ax.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color=color,
        linewidth=3.0 if is_ground else 2.5,
        solid_capstyle="round",
        zorder=1 if is_ground else 2,
    )
    artist.set_gid(f"kimech-body:{body.name}")


def _draw_revolute_joint(config: Configuration, joint: RevoluteJoint, ax) -> None:
    position_a = config.position(joint.point_a)
    position_b = config.position(joint.point_b)
    center = 0.5 * (position_a + position_b)
    artist = ax.scatter(
        [center[0]],
        [center[1]],
        s=58,
        marker="o",
        facecolor="white",
        edgecolor="0.15",
        linewidth=1.5,
        zorder=5,
    )
    artist.set_gid("kimech-joint:revolute")


def _draw_prismatic_joint(
    config: Configuration,
    joint: PrismaticJoint,
    scale: float,
    ax,
) -> None:
    position_a = config.position(joint.point_a)
    position_b = config.position(joint.point_b)
    theta_a = float(config.pose(joint.point_a.body)[2])
    axis = rotation_matrix(theta_a) @ np.asarray(joint.axis_a, dtype=float)
    normal = perpendicular(axis)

    displacement = float(axis @ (position_b - position_a))
    overhang = 0.06 * scale
    start = position_a + (min(0.0, displacement) - overhang) * axis
    end = position_a + (max(0.0, displacement) + overhang) * axis
    (guide,) = ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="0.35",
        linewidth=2.0,
        zorder=3,
    )
    guide.set_gid("kimech-joint:prismatic-guide")

    half_length = 0.045 * scale
    half_width = 0.03 * scale
    vertices = np.asarray(
        [
            position_b - half_length * axis - half_width * normal,
            position_b + half_length * axis - half_width * normal,
            position_b + half_length * axis + half_width * normal,
            position_b - half_length * axis + half_width * normal,
        ]
    )
    slider = Polygon(
        vertices,
        closed=True,
        facecolor="white",
        edgecolor="0.15",
        linewidth=1.5,
        zorder=5,
    )
    slider.set_gid("kimech-joint:prismatic-slider")
    ax.add_patch(slider)