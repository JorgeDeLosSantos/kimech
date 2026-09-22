import pytest

from kimech import (
    Mechanism,
    MechanismTopology,
    PrismaticJoint,
    RevoluteJoint,
)


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

    input_joint = mechanism.revolute(ground_a, crank_a, name="input")
    crank_joint = mechanism.revolute(crank_b, coupler_b, name="crank_coupler")
    coupler_joint = mechanism.revolute(coupler_c, rocker_c, name="coupler_rocker")
    ground_joint = mechanism.revolute(rocker_d, ground_d, name="rocker_ground")

    return (
        mechanism,
        crank,
        coupler,
        rocker,
        input_joint,
        crank_joint,
        coupler_joint,
        ground_joint,
    )


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

    revolute_input = mechanism.revolute(ground_o, crank_o, name="input")
    mechanism.revolute(crank_b, rod_b)
    mechanism.revolute(rod_c, slider_c)
    prismatic = mechanism.prismatic(
        ground_o,
        slider_c,
        axis_a=(1.0, 0.0),
        axis_b=(1.0, 0.0),
        name="slider_guide",
    )
    return mechanism, crank, rod, slider, revolute_input, prismatic


def _watt_two_topology():
    mechanism = Mechanism("watt_ii")
    ground_o2 = mechanism.ground.add_point("O2", (0.0, 0.0))
    ground_o4 = mechanism.ground.add_point("O4", (1.0, 0.0))
    ground_o6 = mechanism.ground.add_point("O6", (2.0, 0.0))

    crank = mechanism.add_link("crank")
    input_rod = mechanism.add_link("input_rod")
    ternary = mechanism.add_link("ternary_rocker")
    second_rod = mechanism.add_link("second_rod")
    output = mechanism.add_link("output_rocker")

    crank_o2 = crank.add_point("O2", (0.0, 0.0))
    crank_a = crank.add_point("A", (1.0, 0.0))
    rod_a = input_rod.add_point("A", (0.0, 0.0))
    rod_b = input_rod.add_point("B", (1.0, 0.0))
    ternary_b = ternary.add_point("B", (0.0, 0.0))
    ternary_o4 = ternary.add_point("O4", (1.0, 0.0))
    ternary_c = ternary.add_point("C", (0.5, 1.0))
    second_c = second_rod.add_point("C", (0.0, 0.0))
    second_d = second_rod.add_point("D", (1.0, 0.0))
    output_d = output.add_point("D", (0.0, 0.0))
    output_o6 = output.add_point("O6", (1.0, 0.0))

    mechanism.revolute(ground_o2, crank_o2)
    mechanism.revolute(crank_a, rod_a)
    mechanism.revolute(rod_b, ternary_b)
    mechanism.revolute(ternary_o4, ground_o4)
    mechanism.revolute(ternary_c, second_c)
    mechanism.revolute(second_d, output_d)
    mechanism.revolute(output_o6, ground_o6)
    return mechanism


def test_topology_preserves_deterministic_body_and_joint_order():
    (
        mechanism,
        crank,
        coupler,
        rocker,
        input_joint,
        crank_joint,
        coupler_joint,
        ground_joint,
    ) = _four_bar()

    topology = mechanism.topology()

    assert isinstance(topology, MechanismTopology)
    assert topology.mechanism is mechanism
    assert topology.bodies == (mechanism.ground, crank, coupler, rocker)
    assert topology.joints == (
        input_joint,
        crank_joint,
        coupler_joint,
        ground_joint,
    )


def test_four_bar_local_topology_queries_and_cycle_rank():
    mechanism, crank, coupler, rocker, *joints = _four_bar()
    topology = mechanism.topology()

    assert topology.incident_joints(crank) == (joints[0], joints[1])
    assert topology.adjacent_bodies(crank) == (mechanism.ground, coupler)
    assert topology.joints_between(crank, coupler) == (joints[1],)
    assert topology.joints_between(coupler, crank) == (joints[1],)
    assert topology.joints_between(crank, crank) == ()
    assert topology.degree(crank) == 2
    assert topology.is_connected
    assert topology.connected_components == (
        (mechanism.ground, crank, coupler, rocker),
    )
    assert topology.cycle_rank == 1


def test_slider_crank_preserves_revolute_and_prismatic_joint_identity():
    mechanism, crank, _, slider, revolute_input, prismatic = _slider_crank()
    topology = mechanism.topology()

    assert isinstance(revolute_input, RevoluteJoint)
    assert isinstance(prismatic, PrismaticJoint)
    assert topology.joints_between(mechanism.ground, crank) == (revolute_input,)
    assert topology.joints_between(mechanism.ground, slider) == (prismatic,)
    assert topology.cycle_rank == 1


def test_parallel_joints_are_preserved_and_count_toward_degree_and_cycle_rank():
    mechanism = Mechanism("parallel")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    ground_b = mechanism.ground.add_point("B", (1.0, 0.0))
    link = mechanism.add_link("link")
    link_a = link.add_point("A", (0.0, 0.0))
    link_b = link.add_point("B", (1.0, 0.0))

    joint_a = mechanism.revolute(ground_a, link_a, name="A")
    joint_b = mechanism.revolute(ground_b, link_b, name="B")
    topology = mechanism.topology()

    assert topology.adjacent_bodies(mechanism.ground) == (link,)
    assert topology.joints_between(mechanism.ground, link) == (joint_a, joint_b)
    assert topology.degree(mechanism.ground) == 2
    assert topology.degree(link) == 2
    assert topology.cycle_rank == 1


def test_disconnected_components_are_deterministic_and_include_isolated_bodies():
    mechanism = Mechanism("disconnected")
    ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
    connected = mechanism.add_link("connected")
    connected_a = connected.add_point("A", (0.0, 0.0))
    orphan = mechanism.add_link("orphan")
    mechanism.revolute(ground_a, connected_a)

    topology = mechanism.topology()

    assert not topology.is_connected
    assert topology.connected_components == (
        (mechanism.ground, connected),
        (orphan,),
    )
    assert topology.cycle_rank == 0


def test_topology_is_a_snapshot_when_mechanism_is_extended():
    mechanism, crank, coupler, rocker, *_ = _four_bar()
    topology = mechanism.topology()

    extension = mechanism.add_link("extension")
    ground_e = mechanism.ground.add_point("E", (1.0, 0.0))
    extension_e = extension.add_point("E", (0.0, 0.0))
    extension_joint = mechanism.revolute(ground_e, extension_e)

    assert topology.bodies == (mechanism.ground, crank, coupler, rocker)
    assert extension not in topology.bodies
    assert extension_joint not in topology.joints
    assert topology.is_connected
    assert topology.cycle_rank == 1

    refreshed = mechanism.topology()
    assert extension in refreshed.bodies
    assert extension_joint in refreshed.joints

    with pytest.raises(ValueError, match="topology snapshot"):
        topology.degree(extension)


def test_body_queries_reject_wrong_types_and_foreign_bodies():
    mechanism, crank, *_ = _four_bar()
    topology = mechanism.topology()

    with pytest.raises(TypeError, match="Ground or Link"):
        topology.degree(object())

    foreign = Mechanism("foreign").add_link("link")
    with pytest.raises(ValueError, match="topology snapshot"):
        topology.degree(foreign)

    assert topology.degree(crank) == 2


def test_watt_two_compound_topology_has_two_independent_cycles():
    mechanism = _watt_two_topology()
    topology = mechanism.topology()

    assert len(topology.bodies) == 6
    assert len(topology.joints) == 7
    assert topology.is_connected
    assert topology.cycle_rank == 2


def test_ground_only_topology_is_connected_and_acyclic():
    mechanism = Mechanism()
    topology = MechanismTopology(mechanism)

    assert topology.bodies == (mechanism.ground,)
    assert topology.joints == ()
    assert topology.connected_components == ((mechanism.ground,),)
    assert topology.is_connected
    assert topology.cycle_rank == 0
