"""Animate solved mechanism configurations with Matplotlib."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ..joints import PrismaticJoint, RevoluteJoint
from ..model import Link, Point
from ..solution import KinematicSolution
from .plot import (
    _auxiliary_connector_coordinates,
    _body_coordinates,
    _body_render_specs,
    _draw_auxiliary_connectors,
    _draw_auxiliary_points,
    _draw_body,
    _draw_prismatic_joint,
    _draw_revolute_joint,
    _link_color_cycle,
    _coordinate_extent,
    _point_positions,
    _scaffold_positions,
    _prismatic_geometry,
    _revolute_center,
)


@dataclass
class _AnimationArtists:
    bodies: dict[Link, tuple[tuple[Point, ...], object]]
    auxiliary: dict[Link, tuple[tuple[Point, ...], object]]
    auxiliary_connectors: dict[Link, tuple[object, object]]
    revolute: dict[RevoluteJoint, object]
    prismatic: dict[PrismaticJoint, tuple[object, object, tuple[float, float]]]
    traces: list[tuple[np.ndarray, object]]


def animate(
    solution: KinematicSolution,
    *,
    fps: float = 30,
    trace_points=None,
    ax=None,
) -> FuncAnimation:
    """Animate a kinematic solution and return Matplotlib's animation object."""
    if not isinstance(solution, KinematicSolution):
        raise TypeError("solution must be a KinematicSolution")
    if len(solution) == 0:
        raise ValueError("solution must contain at least one configuration")
    fps_value = _validate_fps(fps)
    normalized_trace_points = _normalize_trace_points(solution, trace_points)
    trace_data = [
        (point, solution.point_positions(point))
        for point in normalized_trace_points
    ]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    prismatic_ranges = {
        joint: _prismatic_range(solution, joint)
        for joint in solution.mechanism.joints
        if isinstance(joint, PrismaticJoint)
    }
    scale, bounds = _solution_plot_geometry(solution, prismatic_ranges, trace_data)
    artists = _create_artists(solution[0], scale, prismatic_ranges, trace_data, ax)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(index):
        return _update_artists(solution[index], scale, artists, index)

    return FuncAnimation(
        fig,
        update,
        frames=range(len(solution)),
        interval=1000.0 / fps_value,
        blit=False,
        repeat=True,
    )


def _normalize_trace_points(
    solution: KinematicSolution,
    trace_points,
) -> tuple[Point, ...]:
    if trace_points is None:
        return ()
    if isinstance(trace_points, Point):
        raise TypeError("trace_points must be a collection of Point objects")
    try:
        candidates = list(trace_points)
    except TypeError as exc:
        raise TypeError("trace_points must be a collection of Point objects") from exc

    normalized: list[Point] = []
    identities: set[int] = set()
    for point in candidates:
        if not isinstance(point, Point):
            raise TypeError("trace_points must contain only Point objects")
        solution.point_positions(point)
        point_id = id(point)
        if point_id not in identities:
            normalized.append(point)
            identities.add(point_id)
    return tuple(normalized)


def _validate_fps(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Number):
        raise TypeError("fps must be a numeric scalar")
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        raise TypeError("fps must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("fps must be finite")
    if result <= 0.0:
        raise ValueError("fps must be greater than zero")
    return result


def _solution_plot_geometry(
    solution: KinematicSolution,
    prismatic_ranges: dict[PrismaticJoint, tuple[float, float]],
    trace_data: list[tuple[Point, np.ndarray]],
) -> tuple[float, tuple[float, float, float, float]]:
    configurations = [solution[index] for index in range(len(solution))]
    point_positions = np.concatenate([_point_positions(config) for config in configurations])

    scaffold_positions = np.concatenate(
        [
            _scaffold_positions(config, _body_render_specs(config))
            for config in configurations
        ]
    )
    scale = _coordinate_extent(scaffold_positions)
    if scale <= np.finfo(float).eps:
        scale = _coordinate_extent(point_positions)
    if scale <= np.finfo(float).eps:
        scale = 1.0

    rendered_geometry = [point_positions, *[positions for _, positions in trace_data]]
    for config in configurations:
        for joint, guide_range in prismatic_ranges.items():
            start, end, vertices = _prismatic_geometry(
                config,
                joint,
                scale,
                guide_range=guide_range,
            )
            rendered_geometry.append(np.vstack((start, end, vertices)))

    coordinates = np.concatenate(rendered_geometry)
    if len(coordinates) == 0:
        xmin = xmax = ymin = ymax = 0.0
    else:
        xmin, ymin = np.min(coordinates, axis=0)
        xmax, ymax = np.max(coordinates, axis=0)

    margin = 0.16 * scale
    minimum_span = scale
    xcenter = 0.5 * (xmin + xmax)
    ycenter = 0.5 * (ymin + ymax)
    xspan = max(float(xmax - xmin), minimum_span)
    yspan = max(float(ymax - ymin), minimum_span)
    bounds = (
        xcenter - 0.5 * xspan - margin,
        xcenter + 0.5 * xspan + margin,
        ycenter - 0.5 * yspan - margin,
        ycenter + 0.5 * yspan + margin,
    )
    return scale, bounds


def _prismatic_range(
    solution: KinematicSolution,
    joint: PrismaticJoint,
) -> tuple[float, float]:
    values = solution.joint_coordinates(joint)
    return min(0.0, float(values.min())), max(0.0, float(values.max()))


def _create_artists(
    config,
    scale: float,
    prismatic_ranges,
    trace_data: list[tuple[Point, np.ndarray]],
    ax,
) -> _AnimationArtists:
    mechanism = config.mechanism
    specs = _body_render_specs(config)
    body_artists = {}
    auxiliary_artists = {}
    auxiliary_connector_artists = {}
    revolute_artists = {}
    prismatic_artists = {}
    trace_artists = []
    body_colors = {mechanism.ground: "0.4"}

    ground_spec = specs[mechanism.ground]
    _draw_body(config, mechanism.ground, ground_spec.scaffold_points, "0.4", ax)
    _draw_auxiliary_connectors(config, mechanism.ground, ground_spec, "0.4", ax)
    _draw_auxiliary_points(
        config,
        mechanism.ground,
        list(ground_spec.auxiliary_points),
        "0.4",
        ax,
    )

    color_cycle = _link_color_cycle()
    for link in mechanism.links:
        color = next(color_cycle)
        body_colors[link] = color
        spec = specs[link]
        artist = _draw_body(config, link, spec.scaffold_points, color, ax)
        if artist is not None:
            body_artists[link] = (spec.scaffold_points, artist)

        connector_artist = _draw_auxiliary_connectors(config, link, spec, color, ax)
        if connector_artist is not None:
            auxiliary_connector_artists[link] = (spec, connector_artist)

        auxiliary_artist = _draw_auxiliary_points(
            config,
            link,
            list(spec.auxiliary_points),
            color,
            ax,
        )
        if auxiliary_artist is not None:
            auxiliary_artists[link] = (spec.auxiliary_points, auxiliary_artist)

    for joint in mechanism.joints:
        if isinstance(joint, PrismaticJoint):
            guide_range = prismatic_ranges[joint]
            guide, slider = _draw_prismatic_joint(
                config,
                joint,
                scale,
                ax,
                guide_range=guide_range,
            )
            prismatic_artists[joint] = (guide, slider, guide_range)
    for joint in mechanism.joints:
        if isinstance(joint, RevoluteJoint):
            revolute_artists[joint] = _draw_revolute_joint(config, joint, ax)

    for point, positions in trace_data:
        color = body_colors[point.body]
        initial = positions[:1]
        (artist,) = ax.plot(
            initial[:, 0],
            initial[:, 1],
            color=color,
            linewidth=1.2,
            alpha=0.7,
            zorder=1.25,
        )
        artist.set_gid(f"kimech-trace:{point.body.name}:{point.name}")
        trace_artists.append((positions, artist))

    return _AnimationArtists(
        body_artists,
        auxiliary_artists,
        auxiliary_connector_artists,
        revolute_artists,
        prismatic_artists,
        trace_artists,
    )


def _update_artists(
    config,
    scale: float,
    artists: _AnimationArtists,
    index: int,
) -> tuple[object, ...]:
    modified = []
    for points, artist in artists.bodies.values():
        coordinates = _body_coordinates(config, points)
        artist.set_data(coordinates[:, 0], coordinates[:, 1])
        modified.append(artist)
    for points, artist in artists.auxiliary.values():
        artist.set_offsets([config.point_position(point) for point in points])
        modified.append(artist)
    for spec, artist in artists.auxiliary_connectors.values():
        coordinates = _auxiliary_connector_coordinates(
            config,
            spec.scaffold_points,
            spec.connector_points,
        )
        artist.set_data(coordinates[:, 0], coordinates[:, 1])
        modified.append(artist)
    for joint, artist in artists.revolute.items():
        artist.set_offsets([_revolute_center(config, joint)])
        modified.append(artist)
    for joint, (guide, slider, guide_range) in artists.prismatic.items():
        start, end, vertices = _prismatic_geometry(
            config,
            joint,
            scale,
            guide_range=guide_range,
        )
        guide.set_data([start[0], end[0]], [start[1], end[1]])
        slider.set_xy(vertices)
        modified.extend((guide, slider))
    for positions, artist in artists.traces:
        visible = positions[: index + 1]
        artist.set_data(visible[:, 0], visible[:, 1])
        modified.append(artist)
    return tuple(modified)