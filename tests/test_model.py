import numpy as np
import pytest

from kimech import Mechanism


def test_mechanism_has_unique_ground_and_mobile_links_only():
    m = Mechanism("demo")
    crank = m.add_link("crank")

    assert m.ground.name == "ground"
    assert m.links == (crank,)
    assert m.ground not in m.links


def test_link_names_are_unique_and_ground_name_is_reserved():
    m = Mechanism()
    m.add_link("crank")

    with pytest.raises(ValueError):
        m.add_link("crank")
    with pytest.raises(ValueError):
        m.add_link("ground")


def test_point_names_are_local_to_each_body():
    m = Mechanism()
    ground_a = m.ground.add_point("A", (0.0, 0.0))
    crank = m.add_link("crank")
    crank_a = crank.add_point("A", (0.0, 0.0))

    assert ground_a is m.ground["A"]
    assert crank_a is crank["A"]
    assert ground_a is not crank_a


def test_point_coordinates_are_validated_and_not_mutable_through_local_property():
    m = Mechanism()
    link = m.add_link("link")
    point = link.add_point("P", (1.0, 2.0))

    local = point.local
    local[0] = 99.0

    np.testing.assert_allclose(point.local, [1.0, 2.0])

    with pytest.raises(ValueError):
        link.add_point("bad", (1.0, 2.0, 3.0))


def test_links_are_identity_hashable_for_initial_guess_mappings():
    m = Mechanism()
    crank = m.add_link("crank")

    guess = {crank: (0.0, 0.0, 0.0)}
    assert guess[crank] == (0.0, 0.0, 0.0)
