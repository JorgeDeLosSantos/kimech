import numpy as np
import pytest

from kimech import Mechanism, PrismaticJoint, RevoluteJoint


def test_revolute_connects_points_on_distinct_bodies():
    m = Mechanism()
    A0 = m.ground.add_point("A", (0.0, 0.0))
    crank = m.add_link("crank")
    A1 = crank.add_point("A", (0.0, 0.0))

    joint = m.revolute(A0, A1, name=" input ")

    assert isinstance(joint, RevoluteJoint)
    assert joint.name == "input"
    assert m.joints == (joint,)


def test_joint_rejects_two_points_on_same_body():
    m = Mechanism()
    link = m.add_link("link")
    A = link.add_point("A", (0.0, 0.0))
    B = link.add_point("B", (1.0, 0.0))

    with pytest.raises(ValueError):
        m.revolute(A, B)


def test_joint_rejects_points_from_different_mechanisms():
    m1 = Mechanism()
    m2 = Mechanism()
    A = m1.ground.add_point("A", (0.0, 0.0))
    link = m2.add_link("link")
    B = link.add_point("B", (0.0, 0.0))

    with pytest.raises(ValueError):
        RevoluteJoint(A, B)


def test_point_can_be_reused_by_multiple_joints():
    m = Mechanism()
    O = m.ground.add_point("O", (0.0, 0.0))
    crank = m.add_link("crank")
    A = crank.add_point("A", (0.0, 0.0))
    B = crank.add_point("B", (1.0, 0.0))
    rod = m.add_link("rod")
    C = rod.add_point("C", (0.0, 0.0))

    j1 = m.revolute(O, A)
    j2 = m.revolute(B, C)

    assert m.joints == (j1, j2)


def test_prismatic_axes_are_normalized():
    m = Mechanism()
    G = m.ground.add_point("G", (0.0, 0.0))
    slider = m.add_link("slider")
    S = slider.add_point("S", (0.0, 0.0))

    joint = m.prismatic(G, S, axis_a=(10.0, 0.0), axis_b=(0.0, 3.0))

    assert isinstance(joint, PrismaticJoint)
    np.testing.assert_allclose(joint.axis_a, [1.0, 0.0])
    np.testing.assert_allclose(joint.axis_b, [0.0, 1.0])


def test_prismatic_rejects_zero_axis():
    m = Mechanism()
    G = m.ground.add_point("G", (0.0, 0.0))
    slider = m.add_link("slider")
    S = slider.add_point("S", (0.0, 0.0))

    with pytest.raises(ValueError):
        m.prismatic(G, S, axis_a=(0.0, 0.0), axis_b=(1.0, 0.0))
