import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from kimech import Configuration, Mechanism, solve
from kimech.visualization import plot


def _artists_with_gid(ax, gid):
    return [artist for artist in ax.get_children() if artist.get_gid() == gid]


def _four_bar():
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
    guess = {
        crank: (0.0, 0.0, 0.8),
        coupler: (0.05, 0.06, 0.2),
        rocker: (0.30, 0.0, 2.2),
    }
    return mechanism, input_joint, auxiliary, guess


def _slider_crank():
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
    guess = {
        crank: (0.0, 0.0, 0.7),
        rod: (0.06, 0.05, -0.2),
        slider: (0.30, 0.0, 0.0),
    }
    return mechanism, input_joint, guess


def test_importing_root_package_does_not_import_matplotlib():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, kimech; raise SystemExit('matplotlib' in sys.modules)",
        ],
        check=False,
    )
    assert result.returncode == 0


def test_plot_returns_figure_and_axes_and_reuses_external_axes():
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("A", (0.0, 0.0))
    link = mechanism.add_link("link")
    moving = link.add_point("A", (0.0, 0.0))
    mechanism.revolute(fixed, moving)
    config = Configuration(mechanism, [0.0, 0.0, 0.5])

    fig, ax = plot(config)
    existing_fig, existing_ax = plt.subplots()
    fig2, returned_ax = plot(config, ax=existing_ax)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert returned_ax is existing_ax
    assert fig2 is existing_fig
    plt.close(fig)
    plt.close(existing_fig)


def test_plot_can_be_saved_as_nonempty_svg(tmp_path):
    mechanism, input_joint, _, guess = _four_bar()
    config = solve(mechanism, input=input_joint, values=0.8, initial_guess=guess)
    fig, _ = plot(config)
    path = tmp_path / "mechanism.svg"

    try:
        fig.savefig(path)

        assert path.exists()
        assert path.stat().st_size > 0
    finally:
        plt.close(fig)


def test_plot_rejects_non_configuration():
    with pytest.raises(TypeError, match="config must be a Configuration"):
        plot(object())


def test_mobile_links_follow_public_matplotlib_color_cycle():
    mechanism = Mechanism()
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_c = mechanism.ground.add_point("C", (2.0, 0.0))
    first = mechanism.add_link("first")
    first_a = first.add_point("A", (0.0, 0.0))
    first_b = first.add_point("B", (1.0, 0.0))
    first.add_point("auxiliary", (0.5, 0.2))
    second = mechanism.add_link("second")
    second_b = second.add_point("B", (0.0, 0.0))
    second_c = second.add_point("C", (1.0, 0.0))
    second.add_point("auxiliary", (0.5, -0.2))
    mechanism.revolute(ground_a, first_a)
    mechanism.revolute(first_b, second_b)
    mechanism.revolute(second_c, ground_c)
    config = Configuration(mechanism, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    with matplotlib.rc_context({"axes.prop_cycle": cycler(color=["red", "green", "blue"])}):
        fig, ax = plot(config)

    assert _artists_with_gid(ax, "kimech-body:first")[0].get_color() == "red"
    assert _artists_with_gid(ax, "kimech-body:second")[0].get_color() == "green"
    np.testing.assert_allclose(
        _artists_with_gid(ax, "kimech-auxiliary:first")[0].get_facecolors(),
        [to_rgba("red")],
    )
    np.testing.assert_allclose(
        _artists_with_gid(ax, "kimech-auxiliary:second")[0].get_facecolors(),
        [to_rgba("green")],
    )
    plt.close(fig)


def test_four_bar_has_body_skeletons_pivots_and_auxiliary_point():
    mechanism, input_joint, auxiliary, guess = _four_bar()
    config = solve(mechanism, input=input_joint, values=0.8, initial_guess=guess)

    fig, ax = plot(config)

    assert len([line for line in ax.lines if line.get_gid().startswith("kimech-body:")]) == 4
    assert len(_artists_with_gid(ax, "kimech-joint:revolute")) == 4
    auxiliary_artists = _artists_with_gid(ax, "kimech-auxiliary:coupler")
    assert len(auxiliary_artists) == 1
    np.testing.assert_allclose(auxiliary_artists[0].get_offsets(), [config.position(auxiliary)])
    assert ax.get_aspect() == pytest.approx(1.0)
    plt.close(fig)


def test_slider_crank_has_skeleton_guide_slider_patch_and_pivots():
    mechanism, input_joint, guess = _slider_crank()
    config = solve(mechanism, input=input_joint, values=0.7, initial_guess=guess)

    fig, ax = plot(config)

    assert len([line for line in ax.lines if line.get_gid().startswith("kimech-body:")]) >= 2
    guides = _artists_with_gid(ax, "kimech-joint:prismatic-guide")
    sliders = _artists_with_gid(ax, "kimech-joint:prismatic-slider")
    assert len(guides) == 1
    assert np.ptp(guides[0].get_xdata()) > 0.0
    assert len(sliders) == 1
    assert isinstance(sliders[0], Polygon)
    assert len(_artists_with_gid(ax, "kimech-joint:revolute")) == 3
    plt.close(fig)


def test_body_with_three_structural_points_uses_one_hub_and_spoke_line():
    mechanism = Mechanism()
    center = mechanism.add_link("center")
    center_points = [center.add_point(name, coordinates) for name, coordinates in (
        ("A", (0.0, 0.0)),
        ("B", (2.0, 0.0)),
        ("C", (0.0, 1.0)),
    )]
    supports = []
    for index, center_point in enumerate(center_points):
        support = mechanism.add_link(f"support_{index}")
        support_point = support.add_point("P", (0.0, 0.0))
        mechanism.revolute(center_point, support_point)
        supports.append(support)
    config = Configuration(mechanism, np.zeros(3 * len(mechanism.links)))

    fig, ax = plot(config)

    body_lines = _artists_with_gid(ax, "kimech-body:center")
    assert len(body_lines) == 1
    x_data = np.asarray(body_lines[0].get_xdata())
    y_data = np.asarray(body_lines[0].get_ydata())
    assert len(x_data) == 9
    assert np.count_nonzero(np.isnan(x_data)) == 3
    np.testing.assert_allclose(
        np.column_stack((x_data, y_data))[[0, 3, 6]],
        np.tile(np.mean([config.position(point) for point in center_points], axis=0), (3, 1)),
    )
    plt.close(fig)


def test_body_with_one_structural_point_has_joint_but_no_skeleton():
    mechanism = Mechanism()
    fixed = mechanism.ground.add_point("A", (0.0, 0.0))
    link = mechanism.add_link("single")
    moving = link.add_point("A", (0.0, 0.0))
    mechanism.revolute(fixed, moving)
    config = Configuration(mechanism, [0.0, 0.0, 0.3])

    fig, ax = plot(config)

    assert not _artists_with_gid(ax, "kimech-body:single")
    assert len(_artists_with_gid(ax, "kimech-joint:revolute")) == 1
    plt.close(fig)