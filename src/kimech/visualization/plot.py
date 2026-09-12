"""Render solved mechanism configurations as kinematic schematics."""

from __future__ import annotations

from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt, rcParams
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

    color_cycle = _link_color_cycle()
    for link in mechanism.links:
        color = next(color_cycle)
        body_colors[link] = color
        _draw_body(config, link, structural[link], color, ax)

    for joint in mechanism.joints:
        if isinstance(joint, PrismaticJoint):
            _draw_prismatic_joint(config, joint, scale, ax)

    for body in (mechanism.ground, *mechanism.links):
        auxiliary = _auxiliary_points(body, structural[body])
        _draw_auxiliary_points(config, body, auxiliary, body_colors[body], ax)

    for joint in mechanism.joints:
        if isinstance(joint, RevoluteJoint):
            _draw_revolute_joint(config, joint, ax)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def _link_color_cycle():
    colors = rcParams["axes.prop_cycle"].by_key().get("color") or ["C0"]
    return cycle(colors)


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
    coordinates = _point_positions(config)
    if len(coordinates) == 0:
        return 1.0
    extent = float(max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1])))
    return 1.0 if extent <= np.finfo(float).eps else extent


def _point_positions(config: Configuration) -> np.ndarray:
    mechanism = config.mechanism
    positions = [
        config.position(point)
        for body in (mechanism.ground, *mechanism.links)
        for point in body.points
    ]
    return np.asarray(positions, dtype=float).reshape((-1, 2))


def _auxiliary_points(body: _Body, structural: tuple[list[Point], set[int]]) -> list[Point]:
    _, structural_identities = structural
    return [point for point in body.points if id(point) not in structural_identities]


def _body_coordinates(config: Configuration, points: list[Point]) -> np.ndarray:
    positions = np.asarray([config.position(point) for point in points])
    if len(points) == 2:
        return positions
    hub = np.mean(positions, axis=0)
    return np.asarray(
        [coordinate for point in positions for coordinate in (hub, point, (np.nan, np.nan))]
    )


def _revolute_center(config: Configuration, joint: RevoluteJoint) -> np.ndarray:
    position_a = config.position(joint.point_a)
    position_b = config.position(joint.point_b)
    return 0.5 * (position_a + position_b)


def _prismatic_geometry(
    config: Configuration,
    joint: PrismaticJoint,
    scale: float,
    *,
    guide_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_a = config.position(joint.point_a)
    position_b = config.position(joint.point_b)
    theta_a = float(config.body_pose(joint.point_a.body)[2])
    axis = rotation_matrix(theta_a) @ np.asarray(joint.axis_a, dtype=float)
    normal = perpendicular(axis)

    if guide_range is None:
        displacement = float(axis @ (position_b - position_a))
        guide_range = (min(0.0, displacement), max(0.0, displacement))
    guide_min, guide_max = guide_range
    overhang = 0.06 * scale
    start = position_a + (guide_min - overhang) * axis
    end = position_a + (guide_max + overhang) * axis

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
    return start, end, vertices


def _draw_body(
    config: Configuration,
    body: _Body,
    structural: tuple[list[Point], set[int]],
    color: str,
    ax,
):
    points, _ = structural
    if len(points) < 2:
        return None

    coordinates = _body_coordinates(config, points)

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
    return artist


def _draw_auxiliary_points(
    config: Configuration,
    body: _Body,
    points: list[Point],
    color: str,
    ax,
):
    if not points:
        return None
    positions = np.asarray([config.position(point) for point in points])
    artist = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=22,
        color=color,
        zorder=4,
    )
    artist.set_gid(f"kimech-auxiliary:{body.name}")
    return artist


def _draw_revolute_joint(config: Configuration, joint: RevoluteJoint, ax):
    center = _revolute_center(config, joint)
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
    return artist


def _draw_prismatic_joint(
    config: Configuration,
    joint: PrismaticJoint,
    scale: float,
    ax,
    *,
    guide_range: tuple[float, float] | None = None,
):
    start, end, vertices = _prismatic_geometry(
        config,
        joint,
        scale,
        guide_range=guide_range,
    )
    (guide,) = ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="0.35",
        linewidth=2.0,
        zorder=3,
    )
    guide.set_gid("kimech-joint:prismatic-guide")

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
    return guide, slider
