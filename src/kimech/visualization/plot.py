"""Render solved mechanism configurations as kinematic schematics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt, rcParams
from matplotlib.patches import Polygon

from .._geometry import perpendicular, rotation_matrix
from ..joints import PrismaticJoint, RevoluteJoint
from ..model import Ground, Link, Point
from ..solution import Configuration

_Body = Link | Ground


@dataclass(frozen=True)
class _BodyRenderSpec:
    scaffold_points: tuple[Point, ...]
    auxiliary_points: tuple[Point, ...]
    connector_points: tuple[Point, ...]


def plot(config: Configuration, *, ax=None):
    """Plot one solved configuration and return its Matplotlib figure and axes."""
    if not isinstance(config, Configuration):
        raise TypeError("config must be a Configuration")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    mechanism = config.mechanism
    specs = _body_render_specs(config)
    scale = _plot_scale(config, specs)
    body_colors: dict[_Body, str] = {mechanism.ground: "0.4"}

    ground_spec = specs[mechanism.ground]
    _draw_body(config, mechanism.ground, ground_spec.scaffold_points, "0.4", ax)
    _draw_auxiliary_connectors(config, mechanism.ground, ground_spec, "0.4", ax)

    color_cycle = _link_color_cycle()
    for link in mechanism.links:
        color = next(color_cycle)
        body_colors[link] = color
        spec = specs[link]
        _draw_body(config, link, spec.scaffold_points, color, ax)
        _draw_auxiliary_connectors(config, link, spec, color, ax)

    for joint in mechanism.joints:
        if isinstance(joint, PrismaticJoint):
            _draw_prismatic_joint(config, joint, scale, ax)

    for body in (mechanism.ground, *mechanism.links):
        _draw_auxiliary_points(
            config,
            body,
            list(specs[body].auxiliary_points),
            body_colors[body],
            ax,
        )

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


def _body_render_specs(config: Configuration) -> dict[_Body, _BodyRenderSpec]:
    mechanism = config.mechanism
    bodies = (mechanism.ground, *mechanism.links)
    structural: dict[_Body, list[Point]] = {body: [] for body in bodies}
    identities: dict[_Body, set[int]] = {body: set() for body in bodies}

    for joint in mechanism.joints:
        for point in (joint.point_a, joint.point_b):
            point_id = id(point)
            if point_id not in identities[point.body]:
                structural[point.body].append(point)
                identities[point.body].add(point_id)

    specs: dict[_Body, _BodyRenderSpec] = {}
    for body in bodies:
        structural_points = tuple(structural[body])
        auxiliary_points = tuple(
            point for point in body.points if id(point) not in identities[body]
        )

        if isinstance(body, Ground):
            scaffold_points = structural_points
            connector_points: tuple[Point, ...] = ()
        elif len(structural_points) >= 2:
            scaffold_points = structural_points
            connector_points = auxiliary_points
        else:
            scaffold_points = tuple(body.points)
            connector_points = ()

        specs[body] = _BodyRenderSpec(
            scaffold_points=scaffold_points,
            auxiliary_points=auxiliary_points,
            connector_points=connector_points,
        )
    return specs


def _plot_scale(
    config: Configuration,
    specs: dict[_Body, _BodyRenderSpec] | None = None,
) -> float:
    if specs is None:
        specs = _body_render_specs(config)

    scaffold_positions = _scaffold_positions(config, specs)
    extent = _coordinate_extent(scaffold_positions)
    if extent > np.finfo(float).eps:
        return extent

    extent = _coordinate_extent(_point_positions(config))
    return 1.0 if extent <= np.finfo(float).eps else extent


def _scaffold_positions(
    config: Configuration,
    specs: dict[_Body, _BodyRenderSpec],
) -> np.ndarray:
    positions = [
        config.point_position(point)
        for spec in specs.values()
        for point in spec.scaffold_points
    ]
    return np.asarray(positions, dtype=float).reshape((-1, 2))


def _coordinate_extent(coordinates: np.ndarray) -> float:
    if len(coordinates) == 0:
        return 0.0
    return float(max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1])))


def _point_positions(config: Configuration) -> np.ndarray:
    mechanism = config.mechanism
    positions = [
        config.point_position(point)
        for body in (mechanism.ground, *mechanism.links)
        for point in body.points
    ]
    return np.asarray(positions, dtype=float).reshape((-1, 2))


def _body_coordinates(config: Configuration, points: list[Point]) -> np.ndarray:
    positions = np.asarray([config.point_position(point) for point in points])
    if len(points) == 2:
        return positions
    hub = np.mean(positions, axis=0)
    return np.asarray(
        [coordinate for point in positions for coordinate in (hub, point, (np.nan, np.nan))]
    )


def _revolute_center(config: Configuration, joint: RevoluteJoint) -> np.ndarray:
    position_a = config.point_position(joint.point_a)
    position_b = config.point_position(joint.point_b)
    return 0.5 * (position_a + position_b)


def _prismatic_geometry(
    config: Configuration,
    joint: PrismaticJoint,
    scale: float,
    *,
    guide_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_a = config.point_position(joint.point_a)
    position_b = config.point_position(joint.point_b)
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
    points: tuple[Point, ...] | list[Point],
    color: str,
    ax,
):
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


def _auxiliary_connector_coordinates(
    config: Configuration,
    scaffold_points: tuple[Point, ...],
    connector_points: tuple[Point, ...],
) -> np.ndarray:
    if len(scaffold_points) < 2 or not connector_points:
        return np.empty((0, 2), dtype=float)

    scaffold_positions = np.asarray(
        [config.point_position(point) for point in scaffold_points],
        dtype=float,
    )
    if len(scaffold_positions) == 2:
        segments = [(scaffold_positions[0], scaffold_positions[1])]
    else:
        hub = np.mean(scaffold_positions, axis=0)
        segments = [(hub, position) for position in scaffold_positions]

    coordinates: list[np.ndarray | tuple[float, float]] = []
    for point in connector_points:
        position = config.point_position(point)
        anchor = min(
            (_nearest_point_on_segment(position, start, end) for start, end in segments),
            key=lambda candidate: float(np.sum((candidate - position) ** 2)),
        )
        coordinates.extend((anchor, position, (np.nan, np.nan)))
    return np.asarray(coordinates, dtype=float)


def _nearest_point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= np.finfo(float).eps:
        return start.copy()
    parameter = float((point - start) @ delta / denominator)
    parameter = min(1.0, max(0.0, parameter))
    return start + parameter * delta


def _draw_auxiliary_connectors(
    config: Configuration,
    body: _Body,
    spec: _BodyRenderSpec,
    color: str,
    ax,
):
    coordinates = _auxiliary_connector_coordinates(
        config,
        spec.scaffold_points,
        spec.connector_points,
    )
    if len(coordinates) == 0:
        return None
    (artist,) = ax.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color=color,
        linewidth=1.0,
        alpha=0.55,
        solid_capstyle="round",
        zorder=1.5,
    )
    artist.set_gid(f"kimech-auxiliary-connector:{body.name}")
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
    positions = np.asarray([config.point_position(point) for point in points])
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
