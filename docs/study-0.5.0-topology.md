# Kimech 0.5.0 — topology API study

## 1. Purpose

This study evaluates how mechanism topology should become a public concept in
Kimech 0.5.0.

The goal is not to introduce a generic graph library. The goal is to expose the
small amount of structural information that is already implicit in a
`Mechanism` and useful for validation, diagnostics, visualization, future
initial-guess assistance, and later Driver / multi-DOF work.

## 2. Current representation

A mechanism already contains all information needed to construct a topology:

- one `Ground` body;
- an ordered tuple of mobile `Link` bodies;
- an ordered tuple of R/P joints;
- each joint connects exactly two points on two distinct bodies.

Current structural validation builds an internal undirected adjacency mapping
between bodies and performs reachability from ground.

That representation is intentionally small and sufficient for the current
validation task, but it is not a suitable public topology model because a
`set` of adjacent bodies collapses multiple joints between the same pair of
bodies.

A public topology API should preserve joint identity.

## 3. Graph semantics

The natural structural interpretation is an undirected body-joint multigraph:

- vertices: `Ground` and `Link` objects;
- edges: `RevoluteJoint` and `PrismaticJoint` objects;
- each edge joins the two bodies owning its endpoint points.

An equivalent body-joint incidence graph can be constructed when useful, but
the public API does not need to expose graph-theory implementation details.

The important semantic decision is:

> A joint is the primary structural connection. Adjacent bodies alone are not
> enough to represent mechanism topology.

This preserves parallel joints and remains compatible with future joint
families.

## 4. Evidence from current mechanisms

The existing examples and playground models already exercise multiple
independent topological cycles:

| mechanism | bodies including ground | joints | connected components | cycle rank |
| --- | ---: | ---: | ---: | ---: |
| four-bar | 4 | 4 | 1 | 1 |
| slider-crank | 4 | 4 | 1 | 1 |
| Archimedes trammel | 4 | 4 | 1 | 1 |
| Whitworth | 6 | 7 | 1 | 2 |
| Watt II six-bar | 6 | 7 | 1 | 2 |
| Klann | 6 | 7 | 1 | 2 |
| Theo Jansen | 8 | 10 | 1 | 3 |

For an undirected multigraph, the topological cycle rank is

```text
E - V + C
```

where `E` is the number of joints, `V` the number of bodies, and `C` the
number of connected components.

Cycle rank is well defined. A particular enumerated cycle basis is not unique.

## 5. API alternatives

### Alternative A — methods directly on Mechanism

Example:

```python
mechanism.adjacent_bodies(body)
mechanism.incident_joints(body)
mechanism.connected_components()
mechanism.cycle_rank()
```

Advantages:

- minimal ceremony;
- discoverable from the model object;
- no extra public domain type.

Disadvantages:

- `Mechanism` accumulates analysis responsibilities;
- there is no stable snapshot that can be passed to visualization or later
  analysis code;
- future topology capabilities would continue expanding the core model API;
- topology data and model mutation semantics become less explicit.

This option is convenient but does not preserve Kimech's existing separation
between model declaration and analysis/result objects particularly well.

### Alternative B — dedicated MechanismTopology object

Example:

```python
topology = mechanism.topology()

topology.bodies
topology.joints
topology.incident_joints(crank)
topology.adjacent_bodies(crank)
topology.joints_between(crank, coupler)
topology.degree(crank)
topology.connected_components
topology.is_connected
topology.cycle_rank
```

Advantages:

- gives topology a clear domain meaning without turning Kimech into a generic
  graph package;
- provides a stable object for validation, visualization, diagnostics, and
  future initialization work;
- supports explicit snapshot semantics;
- preserves joint identity and parallel connections;
- can grow modestly without cluttering `Mechanism`;
- does not depend on solver state or a selected driver.

Disadvantages:

- adds one public class;
- requires a small decision about snapshot versus live-view behavior.

This option fits Kimech's current architecture best.

### Alternative C — functions in a topology/analysis module

Example:

```python
from kimech.topology import adjacent_bodies, cycle_rank

adjacent_bodies(mechanism, crank)
cycle_rank(mechanism)
```

Advantages:

- keeps `Mechanism` small;
- simple implementation;
- no extra result class required.

Disadvantages:

- fragmented public surface;
- repeated reconstruction of the same structural data;
- awkward handoff to topology visualization;
- harder to define and document deterministic snapshot behavior;
- less natural foundation for richer future structural analysis.

This option is appropriate for isolated utilities but weaker as the primary
topology abstraction.

## 6. Recommended design

Use a dedicated immutable-from-the-caller's-perspective
`MechanismTopology` snapshot, constructed through a convenience method on
`Mechanism`:

```python
topology = mechanism.topology()
```

The public class may live in `kimech.topology` and be exported from
`kimech`.

This combines discoverability with a clean analysis object:

```text
Mechanism
    |
    +-- topology() --> MechanismTopology
    |
    +-- validate() --> ValidationReport
```

The mechanism remains the declarative owner. The topology object is derived
structural analysis.

## 7. Snapshot semantics

`MechanismTopology` should capture, at construction time:

```python
bodies = (mechanism.ground, *mechanism.links)
joints = mechanism.joints
```

Queries operate on that retained body/joint layout.

If the mechanism is later extended with another link or joint, an already
created topology object does not silently change. Calling
`mechanism.topology()` again produces a new snapshot.

This matches the general snapshot philosophy already used by
`Configuration` and `KinematicSolution` and makes visualization and tests
deterministic.

The snapshot retains references to the public domain objects; it does not clone
links, points, or joints.

## 8. Proposed minimal public API

The first implementation should remain deliberately small.

```python
class MechanismTopology:
    @property
    def mechanism(self) -> Mechanism: ...

    @property
    def bodies(self) -> tuple[Ground | Link, ...]: ...

    @property
    def joints(self) -> tuple[RevoluteJoint | PrismaticJoint, ...]: ...

    def incident_joints(self, body) -> tuple[Joint, ...]: ...
    def adjacent_bodies(self, body) -> tuple[Body, ...]: ...
    def joints_between(self, body_a, body_b) -> tuple[Joint, ...]: ...
    def degree(self, body) -> int: ...

    @property
    def connected_components(self) -> tuple[tuple[Body, ...], ...]: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def cycle_rank(self) -> int: ...
```

### Ordering semantics

All public tuples should be deterministic.

- `bodies`: ground first, then mobile links in mechanism creation order;
- `joints`: mechanism joint creation order;
- `incident_joints(body)`: joint creation order;
- `adjacent_bodies(body)`: unique bodies in topology body order;
- `joints_between(a, b)`: joint creation order;
- each connected component: topology body order;
- components: ordered by the first body they contain.

### Degree semantics

`degree(body)` counts incident joints, not unique neighboring bodies.

This is the correct multigraph meaning and preserves the effect of parallel
connections.

## 9. Concepts to keep private or defer

The initial public API should not expose:

- a NetworkX graph object;
- adjacency matrices;
- incidence matrices;
- a generic node/edge wrapper hierarchy;
- a canonical list of loops;
- a particular cycle basis;
- graph traversal algorithms;
- graph layout coordinates;
- a topological singularity classification.

These can be added later if a concrete Kimech use case requires them.

In particular, `cycle_rank` should be documented as a purely structural graph
quantity. It is not mobility, constraint rank, or a count of kinematic
assembly modes.

## 10. Interaction with validation

The current validation code already reconstructs body adjacency.

After `MechanismTopology` is implemented, validation should be able to reuse
the topology representation for connectivity queries, reducing duplicate graph
logic.

However, validation and topology have different responsibilities:

- topology describes the structural snapshot;
- validation decides whether that snapshot is acceptable for Kimech.

The topology object should therefore not become a container for warnings or
validation policy.

## 11. Interaction with topology visualization

Topology visualization in 0.5-B should consume a `MechanismTopology` object
or obtain one from a `Mechanism`.

It should visualize structural connectivity only.

A layout algorithm may choose arbitrary screen positions for bodies. Those
positions must not be confused with mechanism geometry, point coordinates, or
a solved configuration.

Because joint identity is preserved, multiple joints between the same pair of
bodies can be rendered distinctly if such a model appears.

## 12. Preparation for Driver and multi-DOF

The proposed topology API does not know about a driver.

That is desirable.

A future Driver abstraction can annotate or query the same topology without
changing its structural meaning. Multi-DOF support also does not require a
different graph representation.

This is one reason to prefer bodies and joints as the public structural
entities rather than concepts tied to the current single-input solver.

## 13. Acceptance cases for 0.5-A

Focused tests should cover:

1. deterministic body and joint ordering;
2. four-bar adjacency, incident joints, connectivity, and cycle rank;
3. slider-crank preservation of R/P joint identity;
4. a model with two joints between the same pair of bodies;
5. a disconnected mechanism with more than one connected component;
6. snapshot stability after the source mechanism is extended;
7. rejection of a body from another mechanism/snapshot in body queries;
8. cycle-rank behavior for connected and disconnected multigraphs.

At least one compound mechanism should be checked end-to-end after the focused
tests, for example Watt II or Theo Jansen.

## 14. Design decision

The recommended 0.5-A design is:

> Add a public `MechanismTopology` snapshot, discoverable through
> `Mechanism.topology()`, modeled semantically as a body-joint undirected
> multigraph while keeping graph-library details private.

The initial API should expose local incidence/adjacency, connectivity, degree,
and cycle rank, but should defer loop enumeration and generic graph algorithms.

This gives 0.5.0 a useful structural-analysis foundation with low numerical
risk and no dependency on Driver, multi-DOF, or solver state.
