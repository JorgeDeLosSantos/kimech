import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from kimech import KinematicSolution, Mechanism, solve
from kimech.visualization import animate
from kimech.visualization.animation import _prismatic_range


def _artists_with_gid(ax, gid):
    return [artist for artist in ax.get_children() if artist.get_gid() == gid]


def _line_length(line):
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    return np.hypot(x[1] - x[0], y[1] - y[0])


def _finish(animation):
    # FuncAnimation warns on destruction if no frame was rendered by an event loop.
    animation._draw_was_started = True
    plt.close(animation._fig)


def _four_bar_solution():
    mechanism = Mechanism("four_bar")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_d = mechanism.ground.add_point("D", (0.30, 0.0))

    crank = mechanism.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))
    coupler = mechanism.add_link("coupler")
    coupler_b = coupler.add_point("B", (0.0, 0.0))
    coupler_c = coupler.add_point("C", (0.22, 0.0))
    auxiliary = coupler.add_point("P", (0.10, 0.05))
    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)
    solution = solve(
        mechanism,
        input=input_joint,
        values=np.linspace(0.8, 1.3, 4),
        initial_guess={
            crank: (0.0, 0.0, 0.8),
            coupler: (0.05, 0.06, 0.2),
            rocker: (0.30, 0.0, 2.2),
        },
    )
    return solution, auxiliary


def _slider_crank_solution():
    mechanism = Mechanism("slider_crank")
    origin = mechanism.ground.add_point("O", (0.0, 0.0))
    guide = mechanism.ground.add_point("G", (0.0, 0.0))

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))
    rod = mechanism.add_link("connecting_rod")
    rod_b = rod.add_point("B", (0.0, 0.0))
    rod_c = rod.add_point("C", (0.24, 0.0))
    slider = mechanism.add_link("slider")
    slider_c = slider.add_point("C", (0.0, 0.0))
    slider_guide = slider.add_point("G", (0.0, 0.0))

    input_joint = mechanism.revolute(origin, crank_o, name="input")
    mechanism.revolute(crank_b, rod_b)
    mechanism.revolute(rod_c, slider_c)
    mechanism.prismatic(
        guide,
        slider_guide,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
    )
    return solve(
        mechanism,
        input=input_joint,
        values=np.linspace(0.7, 1.2, 4),
        initial_guess={
            crank: (0.0, 0.0, 0.7),
            rod: (0.06, 0.05, -0.2),
            slider: (0.30, 0.0, 0.0),
        },
    )


def _prismatic_solution(values):
    mechanism = Mechanism("prismatic")
    fixed = mechanism.ground.add_point("A", (0.0, 0.0))
    link = mechanism.add_link("slider")
    moving = link.add_point("B", (0.0, 0.0))
    joint = mechanism.prismatic(
        fixed,
        moving,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
    )
    values = np.asarray(values, dtype=float)
    coordinates = np.column_stack((values, np.zeros((len(values), 2))))
    return KinematicSolution(mechanism, joint, values, coordinates), joint


def _mobile_prismatic_solution():
    mechanism = Mechanism("mobile_prismatic")
    body_a = mechanism.add_link("guide")
    point_a = body_a.add_point("A", (0.0, 0.0))
    body_b = mechanism.add_link("slider")
    point_b = body_b.add_point("B", (0.0, 0.0))
    local_axis = (1.0, 0.0)
    joint = mechanism.prismatic(
        point_a,
        point_b,
        axis_a=local_axis,
        axis_b=local_axis,
    )

    guide_positions = np.asarray(((0.0, 0.0), (4.0, 4.0), (8.0, 8.0)))
    displacements = np.asarray((8.0, 4.0, 1.0))
    theta = np.pi / 4.0
    global_axis = np.asarray((np.cos(theta), np.sin(theta)))
    coordinates = np.empty((len(displacements), 6))
    coordinates[:, :2] = guide_positions
    coordinates[:, 2] = theta
    coordinates[:, 3:5] = guide_positions + displacements[:, np.newaxis] * global_axis
    coordinates[:, 5] = theta
    return KinematicSolution(mechanism, joint, displacements, coordinates)


def test_animate_returns_func_animation_with_uniform_fps_interval():
    solution, _ = _four_bar_solution()

    animation = animate(solution, fps=20)

    assert isinstance(animation, FuncAnimation)
    assert animation.event_source.interval == pytest.approx(50.0)
    _finish(animation)


def test_animation_can_be_saved_as_nonempty_gif_with_pillow(tmp_path):
    solution, _ = _four_bar_solution()
    animation = animate(solution, fps=10)
    path = tmp_path / "mechanism.gif"

    try:
        animation.save(path, writer="pillow")

        assert path.exists()
        assert path.stat().st_size > 0
    finally:
        plt.close(animation._fig)


def test_animate_uses_external_axes_without_clearing_existing_content():
    solution, _ = _four_bar_solution()
    fig, ax = plt.subplots()
    existing = ax.axhline(10.0)
    ax.set_title("Existing title")

    animation = animate(solution, ax=ax)

    assert existing in ax.lines
    assert ax.get_title() == "Existing title"
    assert _artists_with_gid(ax, "kimech-body:crank")
    _finish(animation)


def test_animate_rejects_invalid_solution_and_empty_solution():
    with pytest.raises(TypeError, match="solution must be a KinematicSolution"):
        animate(object())

    solution, _ = _four_bar_solution()
    empty = KinematicSolution(
        solution.mechanism,
        solution.input_joint,
        np.empty(0),
        np.empty((0, solution.coordinates.shape[1])),
    )
    with pytest.raises(ValueError, match="at least one"):
        animate(empty)


@pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf, -np.inf])
def test_animate_rejects_invalid_fps_values(fps):
    solution, _ = _four_bar_solution()
    with pytest.raises(ValueError, match="fps"):
        animate(solution, fps=fps)


@pytest.mark.parametrize("fps", ["30", object(), [30], True, 1 + 2j])
def test_animate_rejects_non_numeric_or_non_real_scalar_fps(fps):
    solution, _ = _four_bar_solution()
    with pytest.raises(TypeError, match="fps"):
        animate(solution, fps=fps)


def test_four_bar_updates_existing_body_auxiliary_and_revolute_artists():
    solution, auxiliary = _four_bar_solution()
    fig, ax = plt.subplots()
    animation = animate(solution, ax=ax)
    body = _artists_with_gid(ax, "kimech-body:coupler")[0]
    auxiliary_artist = _artists_with_gid(ax, "kimech-auxiliary:coupler")[0]
    moving_pivot = _artists_with_gid(ax, "kimech-joint:revolute")[1]
    body_before = np.column_stack((body.get_xdata(), body.get_ydata())).copy()
    auxiliary_before = auxiliary_artist.get_offsets().copy()
    pivot_before = moving_pivot.get_offsets().copy()

    animation._func(3)

    assert _artists_with_gid(ax, "kimech-body:coupler")[0] is body
    assert _artists_with_gid(ax, "kimech-auxiliary:coupler")[0] is auxiliary_artist
    assert _artists_with_gid(ax, "kimech-joint:revolute")[1] is moving_pivot
    assert not np.allclose(np.column_stack((body.get_xdata(), body.get_ydata())), body_before)
    assert not np.allclose(auxiliary_artist.get_offsets(), auxiliary_before)
    assert not np.allclose(moving_pivot.get_offsets(), pivot_before)
    np.testing.assert_allclose(auxiliary_artist.get_offsets(), [solution[3].position(auxiliary)])
    _finish(animation)


def test_slider_crank_updates_existing_glyphs_with_fixed_size_and_orientation():
    solution = _slider_crank_solution()
    fig, ax = plt.subplots()
    animation = animate(solution, ax=ax)
    guide = _artists_with_gid(ax, "kimech-joint:prismatic-guide")[0]
    slider = _artists_with_gid(ax, "kimech-joint:prismatic-slider")[0]
    assert isinstance(slider, Polygon)
    vertices_before = slider.get_xy().copy()
    side_lengths_before = np.linalg.norm(np.diff(vertices_before[:4], axis=0), axis=1)
    initial_guide_length = _line_length(guide)

    for index in range(len(solution)):
        animation._func(index)
        assert _artists_with_gid(ax, "kimech-joint:prismatic-guide")[0] is guide
        assert _line_length(guide) == pytest.approx(initial_guide_length)

    vertices_after = slider.get_xy().copy()
    side_lengths_after = np.linalg.norm(np.diff(vertices_after[:4], axis=0), axis=1)
    assert _artists_with_gid(ax, "kimech-joint:prismatic-slider")[0] is slider
    assert not np.allclose(vertices_after, vertices_before)
    np.testing.assert_allclose(side_lengths_after, side_lengths_before)
    np.testing.assert_allclose(guide.get_ydata(), [0.0, 0.0], atol=1e-9)
    _finish(animation)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], (0.0, 3.0)),
        ([-3.0, -2.0, -1.0], (-3.0, 0.0)),
        ([-2.0, 1.0, 3.0], (-2.0, 3.0)),
    ],
)
def test_prismatic_range_covers_full_solution_and_includes_zero(values, expected):
    solution, joint = _prismatic_solution(values)

    assert _prismatic_range(solution, joint) == expected


def test_viewport_contains_mobile_prismatic_guides_and_sliders_in_every_frame():
    solution = _mobile_prismatic_solution()
    fig, ax = plt.subplots()
    animation = animate(solution, ax=ax)
    guide = _artists_with_gid(ax, "kimech-joint:prismatic-guide")[0]
    slider = _artists_with_gid(ax, "kimech-joint:prismatic-slider")[0]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    initial_length = _line_length(guide)

    for index in range(len(solution)):
        animation._func(index)
        geometry = np.vstack(
            (
                np.column_stack((guide.get_xdata(), guide.get_ydata())),
                slider.get_xy(),
            )
        )
        assert np.all(geometry[:, 0] >= xlim[0])
        assert np.all(geometry[:, 0] <= xlim[1])
        assert np.all(geometry[:, 1] >= ylim[0])
        assert np.all(geometry[:, 1] <= ylim[1])
        assert _line_length(guide) == pytest.approx(initial_length)
        assert ax.get_xlim() == xlim
        assert ax.get_ylim() == ylim

    _finish(animation)


def test_updates_keep_artist_counts_and_viewport_fixed():
    solution = _slider_crank_solution()
    fig, ax = plt.subplots()
    animation = animate(solution, ax=ax)
    counts = (len(ax.lines), len(ax.collections), len(ax.patches))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for index in range(len(solution)):
        animation._func(index)
        assert (len(ax.lines), len(ax.collections), len(ax.patches)) == counts
        assert ax.get_xlim() == xlim
        assert ax.get_ylim() == ylim

    _finish(animation)