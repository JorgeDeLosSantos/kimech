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
    _auxiliary_points,
    _body_coordinates,
    _draw_auxiliary_points,
    _draw_body,
    _draw_prismatic_joint,
    _draw_revolute_joint,
    _link_color_cycle,
    _point_positions,
    _prismatic_geometry,
    _revolute_center,
    _structural_points,
)


@dataclass
class _AnimationArtists:
    bodies: dict[Link, tuple[list[Point], object]]
    auxiliary: dict[Link, tuple[list[Point], object]]
    revolute: dict[RevoluteJoint, object]
    prismatic: dict[PrismaticJoint, tuple[object, object, tuple[float, float]]]


def animate(
    solution: KinematicSolution,
    *,
    fps: float = 30,
    ax=None,
) -> FuncAnimation:
    """Animate a kinematic solution and return Matplotlib's animation object."""
    if not isinstance(solution, KinematicSolution):
        raise TypeError("solution must be a KinematicSolution")
    if len(solution) == 0:
        raise ValueError("solution must contain at least one configuration")
    fps_value = _validate_fps(fps)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    prismatic_ranges = {
        joint: _prismatic_range(solution, joint)
        for joint in solution.mechanism.joints
        if isinstance(joint, PrismaticJoint)
    }
    scale, bounds = _solution_plot_geometry(solution, prismatic_ranges)
    artists = _create_artists(solution[0], scale, prismatic_ranges, ax)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(index):
        return _update_artists(solution[index], scale, artists)

    return FuncAnimation(
        fig,
        update,
        frames=range(len(solution)),
        interval=1000.0 / fps_value,
        blit=False,
        repeat=True,
    )


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
) -> tuple[float, tuple[float, float, float, float]]:
    configurations = [solution[index] for index in range(len(solution))]
    point_positions = np.concatenate([_point_positions(config) for config in configurations])
    if len(point_positions) == 0:
        scale = 1.0
    else:
        point_min = np.min(point_positions, axis=0)
        point_max = np.max(point_positions, axis=0)
        extent = float(max(point_max - point_min))
        scale = 1.0 if extent <= np.finfo(float).eps else extent

    rendered_geometry = [point_positions]
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


def _create_artists(config, scale: float, prismatic_ranges, ax) -> _AnimationArtists:
    mechanism = config.mechanism
    structural = _structural_points(config)
    body_artists = {}
    auxiliary_artists = {}
    revolute_artists = {}
    prismatic_artists = {}

    _draw_body(config, mechanism.ground, structural[mechanism.ground], "0.4", ax)
    color_cycle = _link_color_cycle()
    for link in mechanism.links:
        color = next(color_cycle)
        points = structural[link][0]
        artist = _draw_body(config, link, structural[link], color, ax)
        if artist is not None:
            body_artists[link] = (points, artist)
        auxiliary = _auxiliary_points(link, structural[link])
        auxiliary_artist = _draw_auxiliary_points(config, link, auxiliary, color, ax)
        if auxiliary_artist is not None:
            auxiliary_artists[link] = (auxiliary, auxiliary_artist)

    ground_auxiliary = _auxiliary_points(mechanism.ground, structural[mechanism.ground])
    _draw_auxiliary_points(config, mechanism.ground, ground_auxiliary, "0.4", ax)

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

    return _AnimationArtists(
        body_artists,
        auxiliary_artists,
        revolute_artists,
        prismatic_artists,
    )


def _update_artists(config, scale: float, artists: _AnimationArtists) -> tuple[object, ...]:
    modified = []
    for points, artist in artists.bodies.values():
        coordinates = _body_coordinates(config, points)
        artist.set_data(coordinates[:, 0], coordinates[:, 1])
        modified.append(artist)
    for points, artist in artists.auxiliary.values():
        artist.set_offsets([config.position(point) for point in points])
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
    return tuple(modified)