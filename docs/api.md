# Kimech — Package structure and public API candidate

> Status: candidate API for `0.1.0`.
>
> This document refines the conceptual baseline in `docs/design.md` into a concrete package layout and public interface. The API remains provisional until exercised by the first implementation and acceptance tests.

## 1. Design goal

The package structure should mirror the actual responsibilities already validated by the four-bar and slider-crank walkthroughs, without creating extension points that the MVP does not yet need.

The desired user experience is roughly:

```python
import numpy as np
from kimech import Mechanism, solve
from kimech.visualization import animate

m = Mechanism("four_bar")
# ... declare links, points and joints ...

solution = solve(
    m,
    input=J1,
    values=np.linspace(0.5, 0.5 + 2*np.pi, 361),
    initial_guess=guess,
)

path = solution.point_path(P)
animate(solution)
```

The user should not need to know about coordinate-vector indexing, constraint assembly, SciPy callbacks, or renderer internals.

## 2. Initial source layout

```text
src/
└── kimech/
    ├── __init__.py
    ├── model.py
    ├── joints.py
    ├── solution.py
    ├── solver.py
    ├── validation.py
    ├── errors.py
    ├── _geometry.py
    ├── _constraints.py
    └── visualization/
        ├── __init__.py
        ├── plot.py
        └── animation.py
```

This is intentionally flat. In particular, `solver/`, `analysis/`, `drivers/`, backend registries, and mechanism-specific packages are not justified yet.

Modules should only be split further after real implementation pressure appears.

## 3. Module responsibilities

### `model.py`

Owns the declarative kinematic model:

- `Mechanism`
- `Link`
- `Ground`
- `Point`

Responsibilities:

- construct and own the model topology;
- own local point geometry;
- maintain deterministic creation order;
- provide convenient model-building methods;
- expose read-only/topological queries.

Must not contain:

- numerical solving;
- current poses;
- drawing logic;
- solver tolerances.

### `joints.py`

Owns public joint domain objects:

- `RevoluteJoint`
- `PrismaticJoint`

Responsibilities:

- store the body-point references defining each joint;
- store the local axes required by a prismatic joint;
- expose immutable joint metadata.

The public joint objects do not assemble global residual vectors themselves and do not know coordinate-vector indices.

No public `Joint` inheritance hierarchy is required for `0.1.0`; type unions are sufficient unless implementation evidence later justifies a shared base.

### `_geometry.py`

Small private numerical helpers shared by solver, solution, and constraints, for example:

- planar rotation matrix;
- rotated local vectors;
- perpendicular-vector helper;
- point transformation helper.

This module is private because these functions are implementation details, not a geometry sub-library promised to users.

### `_constraints.py`

Private translation layer from public joint objects to mathematical equations.

Responsibilities:

- revolute residual contribution;
- revolute Jacobian contribution;
- prismatic residual contribution;
- prismatic Jacobian contribution;
- prescribed-coordinate residual/Jacobian contribution.

It should work with an internal coordinate map supplied by the solver and must not introduce mechanism-specific equations.

### `solver.py`

Owns the position-solution workflow.

Responsibilities:

- validate a requested solve problem;
- build the deterministic coordinate map;
- pack a user initial guess into the internal coordinate vector;
- assemble complete residual and Jacobian callbacks;
- invoke the initial SciPy root solver;
- independently verify the final residual;
- perform warm-start continuation across input values;
- construct `Configuration` / `KinematicSolution`;
- raise explicit solve errors.

For `0.1.0`, private helpers such as `_CoordinateMap` should remain inside this module unless they become independently substantial.

### `solution.py`

Owns solver-independent result objects:

- `Configuration`
- `KinematicSolution`
- minimal diagnostic data structures if useful.

Responsibilities:

- expose entity-based state queries;
- transform local points to global positions;
- expose joint coordinates;
- derive point paths and link pose histories;
- provide sequence-like access to configurations.

It must not depend on SciPy solver result types.

### `validation.py`

Owns structural/model validation:

- `ValidationReport`
- mobility estimate;
- consistency checks.

The report should remain deliberately lightweight. `0.1.0` does not need a hierarchy of validation-rule objects.

### `errors.py`

Small public exception surface:

- `KimechError`
- `InvalidModelError`
- `KinematicSolveError`

The exact payload of `KinematicSolveError` remains provisional, but it should eventually expose useful failure context such as the failed input index/value and residual information.

### `visualization/`

Matplotlib-only visualization for `0.1.0`.

`plot.py`:

- schematic rendering of a `Configuration`;
- optional trajectory overlays when relevant.

`animation.py`:

- animation of a `KinematicSolution`;
- GIF export;
- presentation-oriented playback settings such as FPS.

The visualization layer consumes domain/result objects and does not mutate them.

No backend abstraction or registry is introduced yet.

## 4. Public import surface

The core top-level namespace should remain small:

```python
from kimech import (
    Mechanism,
    Link,
    Ground,
    Point,
    RevoluteJoint,
    PrismaticJoint,
    Configuration,
    KinematicSolution,
    ValidationReport,
    KimechError,
    InvalidModelError,
    KinematicSolveError,
    solve,
)
```

Typical users will normally need only:

```python
from kimech import Mechanism, solve
```

The remaining types are public primarily for inspection, type annotations, and advanced use.

Visualization is intentionally imported separately:

```python
from kimech.visualization import plot, animate
```

This keeps the core namespace focused and preserves a clean dependency boundary between kinematics and rendering.

Private modules beginning with `_` are not part of the compatibility promise.

## 5. `Mechanism` candidate API

```python
m = Mechanism(name="four_bar")
```

Candidate interface:

```python
class Mechanism:
    @property
    def name(self) -> str | None: ...

    @property
    def ground(self) -> Ground: ...

    @property
    def links(self) -> tuple[Link, ...]: ...

    @property
    def joints(self) -> tuple[RevoluteJoint | PrismaticJoint, ...]: ...

    def add_link(self, name: str) -> Link: ...

    def revolute(
        self,
        point_a: Point,
        point_b: Point,
        *,
        name: str | None = None,
    ) -> RevoluteJoint: ...

    def prismatic(
        self,
        point_a: Point,
        point_b: Point,
        *,
        axis_a: ArrayLike,
        axis_b: ArrayLike,
        name: str | None = None,
    ) -> PrismaticJoint: ...

    def mobility(self) -> int: ...

    def validate(self) -> ValidationReport: ...
```

For `0.1.0`, both `axis_a` and `axis_b` are explicit and required. This is slightly verbose but mathematically unambiguous and does not privilege the local x-axis of either body. Convenience defaults/aliases may be added later only if real usage justifies them.

`links` and `joints` should be exposed as read-only sequences rather than mutable implementation lists.

## 6. `Link`, `Ground`, and `Point`

Candidate use:

```python
crank = m.add_link("crank")
A = crank.add_point("A", (0.0, 0.0))
B = crank.add_point("B", (0.08, 0.0))

same_B = crank["B"]
```

Candidate `Link` interface:

```python
class Link:
    @property
    def name(self) -> str: ...

    @property
    def points(self) -> tuple[Point, ...]: ...

    def add_point(self, name: str, coordinates: ArrayLike) -> Point: ...

    def __getitem__(self, name: str) -> Point: ...
```

`Ground` should offer the same point-owning user interface:

```python
ground.add_point(...)
ground["A"]
```

The exact internal code-sharing strategy between `Link` and `Ground` is deliberately not specified yet. A public `RigidBody` base class should not be introduced merely to eliminate a few lines of duplication.

Candidate `Point` interface:

```python
class Point:
    @property
    def name(self) -> str: ...

    @property
    def body(self) -> Link | Ground: ...

    @property
    def local(self) -> np.ndarray: ...  # shape (2,)
```

Point local coordinates should behave as immutable model geometry after construction.

## 7. Joint candidate API

Joint objects are normally created through `Mechanism`, not instantiated directly by beginner-facing examples.

Candidate public state:

```python
class RevoluteJoint:
    @property
    def name(self) -> str | None: ...

    @property
    def point_a(self) -> Point: ...

    @property
    def point_b(self) -> Point: ...


class PrismaticJoint:
    @property
    def name(self) -> str | None: ...

    @property
    def point_a(self) -> Point: ...

    @property
    def point_b(self) -> Point: ...

    @property
    def axis_a(self) -> np.ndarray: ...

    @property
    def axis_b(self) -> np.ndarray: ...
```

Axes are normalized when the joint is constructed. Zero-length axes are rejected.

Joint coordinate evaluation should normally happen through a `Configuration` or `KinematicSolution`, rather than requiring the joint to know global state.

## 8. `solve()` candidate API

The high-level position solver remains a function:

```python
def solve(
    mechanism: Mechanism,
    *,
    input: RevoluteJoint | PrismaticJoint,
    values: float | ArrayLike,
    initial_guess: Mapping[Link, ArrayLike] | Configuration,
) -> Configuration | KinematicSolution:
    ...
```

Semantics:

- scalar `values` → `Configuration`;
- one-dimensional sequence `values` → `KinematicSolution`;
- `initial_guess` mapping must contain every mobile link;
- mapping poses are `(x, y, theta)`;
- a `Configuration` guess must belong to the same mechanism;
- sequence order is preserved exactly;
- continuation uses the previous accepted configuration as the next guess;
- the first failure aborts the sweep and raises `KinematicSolveError`.

Solver-specific SciPy options are intentionally not exposed in this first public signature. We should add public tolerances/options only when implementation or acceptance tests demonstrate a real need.

## 9. `Configuration` candidate API

```python
class Configuration:
    @property
    def mechanism(self) -> Mechanism: ...

    @property
    def input_value(self) -> float: ...

    @property
    def coordinates(self) -> np.ndarray: ...

    def pose(self, link: Link | Ground) -> np.ndarray: ...

    def position(self, point: Point) -> np.ndarray: ...

    def joint_coordinate(
        self,
        joint: RevoluteJoint | PrismaticJoint,
    ) -> float: ...
```

Shapes:

- `pose(link)` → `(3,)` containing `(x, y, theta)`;
- `position(point)` → `(2,)`;
- `coordinates` → `(3*n,)` for mobile links only.

`Ground` pose is always `(0, 0, 0)`.

The returned raw coordinates should not permit accidental mutation of the accepted internal state; implementation may return a read-only view or copy.

## 10. `KinematicSolution` candidate API

```python
class KinematicSolution:
    @property
    def mechanism(self) -> Mechanism: ...

    @property
    def input_joint(self) -> RevoluteJoint | PrismaticJoint: ...

    @property
    def input_values(self) -> np.ndarray: ...

    @property
    def coordinates(self) -> np.ndarray: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Configuration: ...

    def point_path(self, point: Point) -> np.ndarray: ...

    def link_poses(self, link: Link | Ground) -> np.ndarray: ...

    def joint_coordinates(
        self,
        joint: RevoluteJoint | PrismaticJoint,
    ) -> np.ndarray: ...
```

Expected shapes:

- `input_values` → `(N,)`;
- `coordinates` → `(N, 3*n)`;
- `point_path(point)` → `(N, 2)`;
- `link_poses(link)` → `(N, 3)`;
- `joint_coordinates(joint)` → `(N,)`.

No pandas dependency is introduced. NumPy arrays are the natural numerical interchange format.

## 11. Validation candidate API

```python
report = m.validate()
```

Minimal report:

```python
class ValidationReport:
    @property
    def is_valid(self) -> bool: ...

    @property
    def mobility(self) -> int: ...

    @property
    def errors(self) -> tuple[str, ...]: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...
```

No `ValidationRule`, severity enum, issue-code registry, or visitor framework is justified for the MVP.

`solve()` should refuse to solve a structurally invalid model and raise `InvalidModelError` with or from the report.

## 12. Visualization candidate API

Keep it functional and small:

```python
from kimech.visualization import plot, animate

fig, ax = plot(config)
animation = animate(solution, fps=30)
```

Candidate behavior:

```python
plot(
    config,
    *,
    traces=None,
    ax=None,
)
```

```python
animate(
    solution,
    *,
    traces=None,
    fps=30,
)
```

Export should initially reuse the objects returned by Matplotlib where practical rather than introducing a dedicated export service. Convenience wrappers may be added only if they materially improve common presentation workflows.

## 13. What is deliberately absent

There is no initial public:

- `Driver` class;
- `JointCoordinate` class;
- `PositionSolver` class;
- `SolverOptions` hierarchy;
- `Constraint` class hierarchy;
- `Backend` interface;
- `Renderer` factory;
- `Analysis` object;
- `FourBar` class;
- `SliderCrank` class;
- mutable current-pose state on a mechanism or link.

These omissions are intentional.

## 14. Tests should mirror public responsibilities

Candidate initial test layout:

```text
tests/
├── test_model.py
├── test_joints.py
├── test_validation.py
├── test_solution.py
├── test_four_bar.py
├── test_slider_crank.py
└── test_visualization.py
```

The four-bar and slider-crank tests are integration/acceptance tests of the generic infrastructure, not tests of mechanism-specific implementations.

## 15. Examples

Keep examples separate from the core:

```text
examples/
├── four_bar.py
└── slider_crank.py
```

These examples should build mechanisms entirely through the generic public API. They are also useful as informal API usability tests.

## 16. New implementation decisions

The following decisions refine `docs/design.md` for the initial implementation:

- **D55.** Start with a flat core package; do not create solver/analysis/backend subpackages before they are needed.
- **D56.** `model.py` owns `Mechanism`, `Link`, `Ground`, and `Point`; `joints.py` owns revolute/prismatic domain objects.
- **D57.** Constraint equations live in private `_constraints.py`; shared planar transform helpers live in private `_geometry.py`.
- **D58.** No public common `Joint` or `RigidBody` base class is required for `0.1.0`.
- **D59.** `solve()` remains the single high-level solver entry point; no public `PositionSolver` abstraction is introduced initially.
- **D60.** `axis_a` and `axis_b` are explicit required arguments of the initial prismatic API; convenience defaults are deferred.
- **D61.** The top-level `kimech` namespace exposes core model/result/error types and `solve`, while plotting and animation remain under `kimech.visualization`.
- **D62.** Structural validation returns one lightweight `ValidationReport`; no rule-object hierarchy is introduced.
- **D63.** Raw numerical interchange uses NumPy arrays; pandas is not a dependency.
- **D64.** Solver-specific SciPy options are not exposed in the first public `solve()` signature unless implementation evidence shows they are necessary.
- **D65.** The first implementation keeps coordinate-map helpers private inside `solver.py`; extract them only if they acquire an independent responsibility.
- **D66.** Initial examples are generic API clients, not special mechanism types.

## 17. Remaining API questions

The following should be resolved during the first implementation rather than by more abstract planning:

1. whether raw array properties return copies or read-only views;
2. exact diagnostic payload of `Configuration`, `KinematicSolution`, and `KinematicSolveError`;
3. whether plotting functions return `(fig, ax)` or only the primary Matplotlib object;
4. whether visualization dependencies are normal package dependencies or an optional `viz` extra;
5. whether a characteristic-length override needs to be public in `0.1.0`;
6. whether scalar/sequence-dependent return types from `solve()` remain pleasant once typed and tested.

These are implementation-level questions and should not trigger new architectural layers unless actual code demonstrates the need.
