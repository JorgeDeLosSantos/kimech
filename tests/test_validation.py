from kimech import Mechanism, ValidationReport


def _four_bar():
    m = Mechanism("four_bar")
    A0 = m.ground.add_point("A", (0.0, 0.0))
    D0 = m.ground.add_point("D", (0.30, 0.0))

    crank = m.add_link("crank")
    A1 = crank.add_point("A", (0.0, 0.0))
    B1 = crank.add_point("B", (0.08, 0.0))

    coupler = m.add_link("coupler")
    B2 = coupler.add_point("B", (0.0, 0.0))
    C2 = coupler.add_point("C", (0.22, 0.0))

    rocker = m.add_link("rocker")
    C3 = rocker.add_point("C", (0.0, 0.0))
    D3 = rocker.add_point("D", (0.18, 0.0))

    m.revolute(A0, A1)
    m.revolute(B1, B2)
    m.revolute(C2, C3)
    m.revolute(D3, D0)
    return m


def _slider_crank():
    m = Mechanism("slider_crank")
    O0 = m.ground.add_point("O", (0.0, 0.0))

    crank = m.add_link("crank")
    O1 = crank.add_point("O", (0.0, 0.0))
    B1 = crank.add_point("B", (0.08, 0.0))

    rod = m.add_link("rod")
    B2 = rod.add_point("B", (0.0, 0.0))
    C2 = rod.add_point("C", (0.24, 0.0))

    slider = m.add_link("slider")
    C3 = slider.add_point("C", (0.0, 0.0))

    m.revolute(O0, O1)
    m.revolute(B1, B2)
    m.revolute(C2, C3)
    m.prismatic(O0, C3, axis_a=(1.0, 0.0), axis_b=(1.0, 0.0))
    return m


def test_four_bar_is_connected_and_has_unit_mobility():
    m = _four_bar()

    report = m.validate()

    assert isinstance(report, ValidationReport)
    assert report.is_valid
    assert report.mobility == 1
    assert m.mobility() == 1
    assert report.errors == ()


def test_slider_crank_is_connected_and_has_unit_mobility():
    m = _slider_crank()

    report = m.validate()

    assert report.is_valid
    assert report.mobility == 1
    assert report.errors == ()


def test_disconnected_mobile_link_is_structural_error():
    m = Mechanism()
    m.add_link("orphan")

    report = m.validate()

    assert not report.is_valid
    assert report.mobility == 3
    assert report.errors == ("mobile links disconnected from ground: 'orphan'",)


def test_ground_only_model_is_consistent_but_warns_that_it_has_no_mobile_links():
    m = Mechanism()

    report = m.validate()

    assert report.is_valid
    assert report.mobility == 0
    assert report.warnings == ("mechanism has no mobile links",)


def test_negative_structural_mobility_is_warning_not_automatic_invalidity():
    m = Mechanism()
    A0 = m.ground.add_point("A", (0.0, 0.0))
    B0 = m.ground.add_point("B", (1.0, 0.0))
    link = m.add_link("link")
    A1 = link.add_point("A", (0.0, 0.0))
    B1 = link.add_point("B", (1.0, 0.0))

    m.revolute(A0, A1)
    m.revolute(B0, B1)

    report = m.validate()

    assert report.is_valid
    assert report.mobility == -1
    assert "negative" in report.warnings[0]
