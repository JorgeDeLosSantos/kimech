# Kimech 0.5.0 — planning baseline

## 1. Purpose

Kimech 0.5.0 is planned as a consolidation release for the existing one-DOF
planar R/P kinematics core.

The main goal is to improve introspection, diagnostics, failure semantics, and
analysis capabilities before introducing larger architectural generalizations
such as a public Driver abstraction or multi-DOF solving.

The release should preserve the current scope:

- planar rigid-body mechanisms;
- revolute and prismatic joints;
- structural mobility 1 for the public solver;
- one prescribed natural joint coordinate;
- position, velocity, and acceleration kinematics;
- local continuation on the assembly branch selected by the initial guess.

## 2. Baseline inherited from 0.4.0

Kimech 0.4.0 already provides:

- declarative Mechanism / Link / Point modeling;
- revolute and prismatic joints;
- structural validation and mobility estimation;
- dimensionless numerical scaling;
- position solving with predictor-corrector continuation;
- warm-start fallback and bounded adaptive subdivision;
- analytic velocity and acceleration kinematics;
- Configuration and KinematicSolution query APIs;
- scaled-Jacobian diagnostics:
  - condition number;
  - minimum singular value;
  - numerical rank;
- solve-process diagnostics:
  - accepted strategy;
  - subdivision count;
  - nonlinear corrector-attempt count;
  - final scaled residual norm;
- schematic Matplotlib plotting and animation;
- progressive point traces.

0.5.0 should build on this baseline rather than expand the fundamental solver
domain prematurely.

## 3. Release theme

The provisional theme is:

> **1-DOF analysis, introspection, and failure observability**

The release should make it easier to understand:

1. how a mechanism is connected;
2. how the chosen driven problem behaves numerically;
3. how the mechanism state changes with the prescribed coordinate;
4. why a solve failed when a requested state cannot be accepted.

## 4. Candidate work blocks

### 0.5-A — Public mechanism topology

Explore and design a small public topology/introspection API derived from the
mechanism graph already used internally for validation.

Potential capabilities include:

- bodies adjacent through joints;
- joints incident to a body;
- connected components / connectivity information;
- body and joint degree information;
- cycle / loop information where it can be defined unambiguously;
- lightweight topology summaries suitable for diagnostics and future tooling.

The API should remain domain-focused and should not require NetworkX as a core
dependency unless a concrete need appears.

The initial topology study recommends a dedicated public
`MechanismTopology` snapshot, discoverable through `Mechanism.topology()`.
Its structural semantics are those of an undirected body-joint multigraph so
that joint identity and parallel connections are preserved. See
[`study-0.5.0-topology.md`](study-0.5.0-topology.md).

#### Acceptance direction

- topology information is deterministic and immutable from the caller's point
  of view;
- ground and mobile bodies are represented consistently;
- R/P joint identity is preserved;
- standard four-bar, slider-crank, six-bar, Klann, and Theo Jansen models can
  be inspected without mechanism-specific code.

### 0.5-B — Topology visualization

Explore a visualization distinct from solved geometric plotting.

The intended distinction is:

- `plot(configuration)`: solved kinematic geometry;
- topology visualization: structural body/joint connectivity.

A topology plot should not imply physical geometry or link length.

#### Acceptance direction

- useful for simple and compound mechanisms;
- visually distinguishes ground, bodies, revolute joints, and prismatic joints;
- remains optional under the existing visualization dependency group;
- does not couple the solver to Matplotlib.

### 0.5-C — Richer solve diagnostics and structured failures

Extend observability without defining a universal binary singularity policy.

Possible additions include:

- diagnostic summaries over a solution history;
- locations of worst conditioning / minimum singular value;
- maximum subdivision and corrector-attempt information;
- exact rank-loss event locations when present;
- structured failure context attached to `KinematicSolveError`.

Structured solve-failure information may include, when available:

- requested input index;
- requested input position;
- final scaled residual norm;
- attempted continuation/recovery path;
- corrector-attempt count;
- Jacobian metrics for the last valid or failed candidate where meaningful.

Public information should describe Kimech semantics rather than expose
SciPy-specific implementation counters.

#### Semantic constraint

A large condition number must remain descriptive information, not an automatic
library-wide `is_singular` classification.

### 0.5-D — Input-coordinate sensitivity

Study whether the input tangent already computed internally for continuation
should become part of the public analysis surface.

The natural first object is

```text
dq / du
```

for the current single prescribed coordinate `u`.

Possible derived queries include sensitivities of:

- body pose;
- point position;
- joint coordinate.

This work should establish semantics that can later generalize naturally from a
vector tangent in the one-DOF case to a sensitivity matrix if multi-DOF support
is introduced.

#### Acceptance direction

- results are consistent with finite-difference checks away from singular
  driven states;
- scaling remains dimensionally and numerically consistent;
- no artificial singularity threshold is introduced;
- sensitivity evaluation does not alter the accepted position solution.

### 0.5-E — Regression and acceptance expansion

Before larger architectural changes, broaden the permanent acceptance baseline.

Candidate mechanisms / cases:

- four-bar:
  - crank-rocker;
  - double-rocker where a valid local input interval is selected;
- slider-crank with revolute and prismatic driving;
- Archimedes trammel;
- Whitworth quick-return;
- Watt II six-bar;
- Klann linkage;
- Theo Jansen linkage;
- near-toggle / dead-center driver-dependent cases;
- forward/reverse sweeps;
- consistent length-scale transformations.

Not every exploratory mechanism must become a per-commit test. The goal is a
balanced set of permanent acceptance cases plus reproducible studies.

## 5. Exploratory study: initial-guess assistance

Initial-guess construction remains one of the main user-friction points.

0.5.0 may include a study of possible assistance strategies, for example:

- partial initial guesses;
- propagation through topology;
- reuse of a known Configuration;
- bounded multistart strategies;
- mechanism-specific geometric initialization only as research evidence.

This study does **not** imply a commitment to `initial_guess=None`.

Automatic branch discovery and assembly-mode enumeration remain separate,
larger problems.

## 6. Explicitly deferred from 0.5.0

The following are intentionally not baseline requirements for 0.5.0:

- public Driver abstraction;
- multiple simultaneous prescribed inputs;
- general multi-DOF solving;
- motion-law or physical-time abstractions;
- pseudo-arclength continuation;
- global branch enumeration or assembly-mode identifiers;
- dynamics, forces, torques, masses, or inertias;
- new joint families;
- automatic branch-independent initial-guess generation;
- GUI / web application architecture.

These directions remain valid future work but should not destabilize the
one-DOF core during this release.

## 7. Architectural preparation for Driver and multi-DOF

0.5.0 should avoid introducing APIs that make the later Driver and multi-DOF
generalizations harder.

In particular:

- new sensitivity semantics should admit a future matrix-valued generalization;
- topology APIs should not assume exactly one driver;
- structured errors should distinguish mechanism state from the selected
  driven formulation;
- diagnostics should continue to describe the selected solve formulation;
- result objects should not encode additional one-DOF assumptions unless they
  are already part of the existing public contract.

The desired progression is approximately:

```text
robust 1-DOF core
      |
      +-- topology / diagnostics / errors / sensitivity
      |
      v
public Driver abstraction
      |
      +-- motion/time semantics
      |
      v
multiple drivers
      |
      v
general multi-DOF solving
```

The exact release numbers for the later stages are deliberately not fixed here.

## 8. Provisional implementation order

The current proposed order is:

1. topology study and API design;
2. public topology implementation;
3. topology visualization;
4. structured solve-failure design;
5. diagnostic-summary improvements;
6. input-sensitivity study and API decision;
7. acceptance/regression expansion;
8. initial-guess exploratory study;
9. documentation, changelog, and release preparation.

This order is provisional. Each block should be reviewed before becoming part
of the public API.

## 9. Release acceptance principles

Kimech 0.5.0 should be considered complete only if:

- the existing 0.4.0 public solve behavior remains valid unless an intentional
  breaking change is explicitly documented;
- existing 1-DOF position, velocity, and acceleration acceptance cases continue
  to pass;
- new public APIs have mechanism-independent semantics;
- new diagnostics remain scale-aware and do not expose backend-specific policy;
- topology functionality is useful independently of visualization;
- sensitivity results have independent numerical verification;
- no feature requires Driver, multi-DOF, or dynamics to make semantic sense;
- documentation clearly separates structural topology, physical configuration,
  and driven-solve diagnostics.

## 10. Design questions

### Resolved for topology

The initial topology study resolves the first two planning questions:

1. topology should use a dedicated immutable-from-the-caller's-perspective
   `MechanismTopology` snapshot, discoverable through `Mechanism.topology()`;
2. the first public concepts should be body/joint snapshots, incident joints,
   adjacent bodies, joints between bodies, multigraph degree, connected
   components, connectivity, and cycle rank.

Loop enumeration, a canonical cycle basis, matrices, generic traversal APIs,
and graph-library objects remain deferred.

See [`study-0.5.0-topology.md`](study-0.5.0-topology.md).

### Resolved for diagnostics and sensitivity

The 0.5-C implementation uses a dedicated immutable `SolveFailureContext`
attached optionally to `KinematicSolveError`, and
`SolveDiagnostics.summary()` returns a `SolveDiagnosticSummary`.

The 0.5-D sensitivity study resolves the next two questions:

1. `dq/du` is computed explicitly on request through
   `input_sensitivity(solution)`, rather than stored by every solve;
2. the first public sensitivity surface includes generalized-coordinate,
   body-pose, point-position, and natural joint-coordinate derivatives.

See
[`study-0.5.0-input-sensitivity.md`](study-0.5.0-input-sensitivity.md).

### Still open

1. Which complex mechanisms should become permanent acceptance tests rather
   than remain reproducible studies?

These questions define the remaining initial 0.5.0 design phase.
