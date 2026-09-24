import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch

from kimech import Mechanism
from kimech.visualization import plot_topology
from kimech.visualization.topology import _topology_layout


def _artists_with_gid_prefix(ax, prefix):
    return [
        artist
        for artist in ax.get_children()
        if artist.get_gid() is not None and artist.get_gid().startswith(prefix)
    ]


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

    rocker = mechanism.add_link("rocker")
    rocker_c = rocker.add_point("C", (0.0, 0.0))
    rocker_d = rocker.add_point("D", (0.18, 0.0))

    mechanism.revolute(ground_a, crank_a)
    mechanism.revolute(crank_b, coupler_b)
    mechanism.revolute(coupler_c, rocker_c)
    mechanism.revolute(rocker_d, ground_d)
    return mechanism


def _slider_crank():
    mechanism = Mechanism("slider_crank")
    ground_o = mechanism.ground.add_point("O", (0.0, 0.0))

    crank = mechanism.add_link("crank")
    crank_o = crank.add_point("O", (0.0, 0.0))
    crank_b = crank.add_point("B", (0.08, 0.0))

    rod = mechanism.add_link("rod")
    rod_b = rod.add_point("B", (0.0, 0.0))
    rod_c = rod.add_point("C", (0.24, 0.0))

    slider = mechanism.add_link("slider")
    slider_c = slider.add_point("C", (0.0, 0.0))

    mechanism.revolute(ground_o, crank_o)
    mechanism.revolute(crank_b, rod_b)
    mechanism.revolute(rod_c, slider_c)
    mechanism.prismatic(
        ground_o,
        slider_c,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
    )
    return mechanism


def test_plot_topology_accepts_mechanism_and_reuses_external_axes():
    mechanism = _four_bar()

    fig, ax = plot_topology(mechanism)
    existing_fig, existing_ax = plt.subplots()
    fig2, returned_ax = plot_topology(mechanism.topology(), ax=existing_ax)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert returned_ax is existing_ax
    assert fig2 is existing_fig
    plt.close(fig)
    plt.close(existing_fig)


def test_plot_topology_rejects_unrelated_input():
    with pytest.raises(
        TypeError,
        match="Mechanism or MechanismTopology",
    ):
        plot_topology(object())


def test_four_bar_draws_one_node_per_body_and_one_connection_per_joint():
    mechanism = _four_bar()

    fig, ax = plot_topology(mechanism)

    assert len(_artists_with_gid_prefix(ax, "kimech-topology-body:")) == 4
    assert len(_artists_with_gid_prefix(ax, "kimech-topology-edge:")) == 4
    assert len(_artists_with_gid_prefix(ax, "kimech-topology-joint:revolute:")) == 4
    assert len(_artists_with_gid_prefix(ax, "kimech-topology-joint:prismatic:")) == 0
    assert len(_artists_with_gid_prefix(ax, "kimech-topology-label:")) == 4
    assert not ax.axison
    plt.close(fig)


def test_slider_crank_visually_distinguishes_revolute_and_prismatic_joints():
    mechanism = _slider_crank()

    fig, ax = plot_topology(mechanism)

    revolute = _artists_with_gid_prefix(ax, "kimech-topology-joint:revolute:")
    prismatic = _artists_with_gid_prefix(ax, "kimech-topology-joint:prismatic:")

    assert len(revolute) == 3
    assert len(prismatic) == 1

    edges = _artists_with_gid_prefix(ax, "kimech-topology-edge:")
    assert all(isinstance(edge, PathPatch) for edge in edges)
    assert len(edges) == 4
    plt.close(fig)


def test_parallel_joints_get_distinct_curves_without_collapsing_identity():
    mechanism = Mechanism("parallel")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_b = mechanism.ground.add_point("B", (1.0, 0.0))
    link = mechanism.add_link("link")
    link_a = link.add_point("A", (0.0, 0.0))
    link_b = link.add_point("B", (1.0, 0.0))

    mechanism.revolute(ground_a, link_a)
    mechanism.revolute(ground_b, link_b)

    fig, ax = plot_topology(mechanism)

    edges = _artists_with_gid_prefix(ax, "kimech-topology-edge:")
    assert len(edges) == 2
    vertices_a = edges[0].get_path().vertices
    vertices_b = edges[1].get_path().vertices
    assert not np.allclose(vertices_a[1], vertices_b[1])
    plt.close(fig)


def test_layout_is_independent_of_physical_point_coordinates():
    mechanism_a = Mechanism("first")
    mechanism_b = Mechanism("second")

    def populate(mechanism, scale, offset):
        ground_a = mechanism.ground.add_point("A", offset)
        ground_d = mechanism.ground.add_point(
            "D",
            (offset[0] + 0.30 * scale, offset[1]),
        )

        crank = mechanism.add_link("crank")
        crank_a = crank.add_point("A", (0.0, 0.0))
        crank_b = crank.add_point("B", (0.08 * scale, 0.0))

        coupler = mechanism.add_link("coupler")
        coupler_b = coupler.add_point("B", (0.0, 0.0))
        coupler_c = coupler.add_point("C", (0.22 * scale, 0.0))

        rocker = mechanism.add_link("rocker")
        rocker_c = rocker.add_point("C", (0.0, 0.0))
        rocker_d = rocker.add_point("D", (0.18 * scale, 0.0))

        mechanism.revolute(ground_a, crank_a)
        mechanism.revolute(crank_b, coupler_b)
        mechanism.revolute(coupler_c, rocker_c)
        mechanism.revolute(rocker_d, ground_d)

    populate(mechanism_a, 1.0, (0.0, 0.0))
    populate(mechanism_b, 1000.0, (50.0, -20.0))

    topology_a = mechanism_a.topology()
    topology_b = mechanism_b.topology()
    layout_a = _topology_layout(topology_a)
    layout_b = _topology_layout(topology_b)

    positions_a = np.asarray([layout_a[body] for body in topology_a.bodies])
    positions_b = np.asarray([layout_b[body] for body in topology_b.bodies])
    np.testing.assert_allclose(positions_a, positions_b)


def test_disconnected_components_are_spatially_separated():
    mechanism = Mechanism("disconnected")
    connected = mechanism.add_link("connected")
    orphan = mechanism.add_link("orphan")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    connected_a = connected.add_point("A", (0.0, 0.0))
    mechanism.revolute(ground_a, connected_a)

    topology = mechanism.topology()
    layout = _topology_layout(topology)

    connected_center = np.mean(
        [layout[mechanism.ground], layout[connected]],
        axis=0,
    )
    orphan_position = layout[orphan]

    assert np.linalg.norm(orphan_position - connected_center) > 1.0


def test_plot_topology_can_be_saved_as_nonempty_svg(tmp_path):
    mechanism = _slider_crank()
    fig, _ = plot_topology(mechanism)
    path = tmp_path / "topology.svg"

    try:
        fig.savefig(path)
        assert path.exists()
        assert path.stat().st_size > 0
    finally:
        plt.close(fig)
