import numpy as np
import pytest

from kimech import Configuration, KinematicDriver, KinematicSolution, Mechanism
from kimech._drivers import normalize_drivers


def _two_revolute_chain():
    mechanism = Mechanism()
    base = mechanism.ground.add_point("base", (0, 0))
    link1 = mechanism.add_link("link1")
    a = link1.add_point("a", (0, 0))
    b = link1.add_point("b", (1, 0))
    link2 = mechanism.add_link("link2")
    c = link2.add_point("c", (0, 0))
    j1 = mechanism.revolute(base, a)
    j2 = mechanism.revolute(b, c)
    return mechanism, j1, j2


def test_normalizer_accepts_one_or_many_and_preserves_order():
    mech, j1, j2 = _two_revolute_chain()
    a = KinematicDriver(j1, position=[0.1, 0.2], velocity=1.0)
    b = KinematicDriver(j2, position=[0.3, 0.4], velocity=2.0)
    assert normalize_drivers(a) == (a,)
    assert normalize_drivers([a, b]) == (a, b)


def test_normalizer_rejects_invalid_and_misaligned_histories():
    _, j1, j2 = _two_revolute_chain()
    a = KinematicDriver(j1, position=[0.1, 0.2])
    b = KinematicDriver(j2, position=[0.3])
    with pytest.raises(ValueError, match="same number"):
        normalize_drivers([a, b])
    with pytest.raises(ValueError, match="same joint"):
        normalize_drivers([a, KinematicDriver(j1, position=[0.4, 0.5])])
    with pytest.raises(TypeError, match="drivers\\[1\\]"):
        normalize_drivers([a, object()])
    with pytest.raises(ValueError, match="at least one"):
        normalize_drivers([])
    with pytest.raises(TypeError, match="sequence"):
        normalize_drivers("not a driver")


def test_multidof_solution_and_configurations_keep_all_sampled_drivers():
    mech, j1, j2 = _two_revolute_chain()
    a = KinematicDriver(j1, position=[0.1, 0.2, 0.3], velocity=1.0, acceleration=0.0)
    b = KinematicDriver(j2, position=[0.4, 0.5, 0.6], velocity=2.0, acceleration=0.0)
    q = np.zeros((3, 6))
    sol = KinematicSolution(mech, [a, b], q, time=[0.0, 0.1, 0.2])
    assert len(sol) == 3
    assert sol.drivers == (a, b)
    assert tuple(d.position for d in sol[1].drivers) == pytest.approx((0.2, 0.5))
    assert tuple(d.velocity for d in sol[1].drivers) == pytest.approx((1.0, 2.0))
    assert sol[1].time == pytest.approx(0.1)
    subset = sol[::-1]
    assert len(subset.drivers) == 2
    np.testing.assert_allclose(subset.drivers[0].position, [0.3, 0.2, 0.1])
    np.testing.assert_allclose(subset.drivers[1].position, [0.6, 0.5, 0.4])
    np.testing.assert_allclose(subset.time, [0.2, 0.1, 0.0])


def test_configuration_without_prescriptions_and_foreign_joint():
    mech, j1, j2 = _two_revolute_chain()
    empty = Configuration(mech, np.zeros(6))
    assert empty.drivers == ()
    other = Mechanism()
    ground_point = other.ground.add_point("G", (0, 0))
    foreign = other.add_link("foreign")
    other_joint = other.revolute(ground_point, foreign.add_point("F", (0, 0)))
    with pytest.raises(ValueError, match="joint"):
        Configuration(mech, np.zeros(6), drivers=KinematicDriver(other_joint, position=0.0))
    with pytest.raises(ValueError, match="one sample"):
        Configuration(mech, np.zeros(6), drivers=KinematicDriver(j1, position=[0.0, 0.1]))


def test_solution_rejects_mismatched_drivers():
    mech, j1, j2 = _two_revolute_chain()
    with pytest.raises(ValueError, match="same number"):
        KinematicSolution(
            mech,
            [KinematicDriver(j1, position=[0, 1]), KinematicDriver(j2, position=[0])],
            np.zeros((2, 6)),
        )
