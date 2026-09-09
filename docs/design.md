# Kimech — Design baseline

> Status: planning baseline for the initial MVP (`0.1.0`).
>
> This document records the current conceptual design of Kimech. It is intentionally narrower than a general multibody package and should be treated as the source of truth for early implementation decisions.

## 1. Purpose

Kimech is a small Python library for modeling, solving, analyzing, visualizing, and animating **planar rigid-body mechanisms**.

The project originated from the need to create didactic mechanism animations for presentations, but the core is designed so that modeling and kinematic solution remain independent from visualization.

The library should remain small, coherent, and useful. It is **not** intended to become a general-purpose multibody dynamics engine.

## 2. MVP scope (`0.1.0`)

The initial MVP should be able to:

- represent planar rigid links;
- define local points on links and on ground;
- model revolute joints;
- model prismatic joints;
- prescribe one rotational or translational joint coordinate as input;
- solve an isolated mechanism configuration;
- sweep a sequence of input values;
- track a solution branch by warm-start continuation;
- obtain point trajectories;
- query link poses and joint coordinates;
- estimate structural mobility;
- validate basic model consistency;
- visualize a configuration in 2D;
- animate a solved motion;
- export static graphics to SVG;
- export animations to GIF.

The MVP is considered successful when the same generic infrastructure can correctly model, solve, and animate at least:

1. a four-bar linkage;
2. a slider-crank mechanism.

No mechanism-specific solver should be required for either case.

## 3. Explicitly out of scope for the MVP

The following are deliberately excluded from `0.1.0`:

- dynamics;
- forces and torques;
- mass and inertia modeling;
- flexibility and FEM;
- general contact;
- spatial mechanisms;
- CAD geometry;
- mechanism synthesis;
- cam contact;
- gear meshing;
- symbolic equation generation as the core solution strategy;
- automatic enumeration of assembly branches;
- adaptive continuation;
- general plugin systems;
- DSL/YAML/JSON model definitions;
- physical unit packages;
- GUI tooling;
- multiple interchangeable nonlinear solver backends.

Possible future extensions should be enabled by clean boundaries, not implemented prematurely.

## 4. Design philosophy

### 4.1 Separate the model from the solver

A `Mechanism` describes what bodies, points, and joints exist. It does not run Newton iterations, manage tolerances, or store a current configuration.

Conceptually:

```python
solution = solve(mechanism, ...)
```

is preferred over:

```python
mechanism.solve()
```

### 4.2 Separate the solver from visualization

The kinematic solution must be reusable without importing a graphical backend.

```text
Mechanism
    ↓
Solver
    ↓
KinematicSolution
    ↓
Analysis / Plot / Animation / Export
```

### 4.3 Model rigid-body kinematics, not drawing primitives

A link is not a Matplotlib line, SVG path, polygon, or CAD object. Visualization is derived from the solved model and remains a separate layer.

### 4.4 Prefer declarative Python over a custom DSL

The Python API itself should be sufficiently declarative. YAML, JSON, parsers, and serialization formats are not required for the initial design.

### 4.5 Results are domain objects, not anonymous arrays

NumPy arrays may be used internally, but users should normally interact through objects such as `Configuration` and `KinematicSolution`.

### 4.6 Keep simple tasks simple

The public API should favor direct operations such as:

```python
solution = solve(m, input=J1, values=theta, initial_guess=guess)
path = solution.point_path(P)
```

instead of requiring users to manually assemble residual vectors, coordinate maps, or solver objects.

## 5. Conceptual architecture

```text
                    ┌─────────────┐
                    │  Mechanism  │
                    └──────┬──────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
            Links        Joints        Ground
              │            │
            Points         │
              └───────┬────┘
                      │
                      ▼
             Constraint assembly
                      │
                      ▼
               Position solver
                      │
                      ▼
             KinematicSolution
                /      |       \
               /       |        \
              ▼        ▼         ▼
          Analysis    Plot     Animation
```

The important boundary is:

```text
Public domain model     Mathematical infrastructure
-------------------     ---------------------------
Joint               →   constraint equations
Link                →   coordinate mapping
Point               →   transformed point positions
```

Constraint equations are initially an internal mathematical representation, not the public modeling language.

## 6. Domain model

### 6.1 `Mechanism`

`Mechanism` is the aggregate that owns the kinematic model.

It contains:

- exactly one ground;
- zero or more mobile links;
- joints connecting bodies through local geometry.

It does not contain:

- a current configuration;
- a permanent driver;
- solver state;
- visualization state.

Typical construction:

```python
m = Mechanism("four_bar")
crank = m.add_link("crank")
```

### 6.2 `Link`

`Link` is the fundamental mobile rigid body of the public API.

Each link defines:

- an arbitrary local reference frame;
- named local points rigidly attached to that frame.

A mobile planar link has a pose

\[
\mathbf q_i =
\begin{bmatrix}
x_i & y_i & \theta_i
\end{bmatrix}^T.
\]

A local point

\[
\mathbf s_P =
\begin{bmatrix}
s_x & s_y
\end{bmatrix}^T
\]

has global position

\[
\mathbf r_P =
\begin{bmatrix}
x_i\\y_i
\end{bmatrix}
+ R(\theta_i)\mathbf s_P,
\]

where

\[
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}.
\]

The local frame is arbitrary. It need not be centered, aligned with the physical link, or located at a joint.

`Link` deliberately does not contain:

- global pose;
- mass or inertia;
- material;
- graphical shape;
- color or thickness;
- velocity or acceleration.

### 6.3 `Point`

A `Point` is a lightweight entity that:

- belongs to exactly one body (`Link` or ground);
- has a name unique within that body;
- stores two local coordinates;
- does not store its global position.

Example:

```python
P = coupler.add_point("P", (0.10, 0.05))
```

Global position is configuration-dependent:

```python
config.position(P)
```

The same point may be referenced by multiple joints.

### 6.4 `Ground`

Each mechanism has exactly one ground, available directly as:

```python
ground = m.ground
```

Ground:

- defines the global frame;
- can own named points;
- has identity pose permanently;
- does not contribute generalized coordinates.

Multiple fixed pivots or guide references are simply multiple points on the same ground.

For `n` mobile links,

\[
\mathbf q \in \mathbb R^{3n},
\]

not `3(n+1)`.

## 7. Joints

The MVP contains two public joint types:

- `RevoluteJoint`;
- `PrismaticJoint`.

A joint connects exactly two bodies. The order `A → B` is meaningful because it determines the sign/reference of the natural joint coordinate.

A joint does not own points; it references points owned by bodies.

### 7.1 Revolute joint

A revolute joint references one local point on each body and imposes coincidence:

\[
\Phi_R(\mathbf q)
=
\mathbf r_{P_A}-\mathbf r_{P_B}
=
\mathbf 0.
\]

This contributes two scalar constraints.

Candidate API:

```python
J = m.revolute(body_a["A"], body_b["A"], name="input")
```

Its natural scalar coordinate is the relative frame angle

\[
\phi_J = \theta_B-\theta_A.
\]

For a joint against ground, this reduces to the mobile body orientation, subject to joint argument order.

Angular coordinates remain **unwrapped**. They are not forced into intervals such as `[-π, π)`.

### 7.2 Prismatic joint

A planar prismatic joint allows one relative translation and suppresses:

- transverse translation;
- relative rotation.

Conceptually each side supplies:

- a local reference point;
- a local oriented axis.

Let `axis_a` define a unit direction on body A:

\[
\mathbf a_A = R(\theta_A)\mathbf a_A^L.
\]

Define the perpendicular direction

\[
\mathbf n_A =
\begin{bmatrix}
-a_{Ay}\\a_{Ax}
\end{bmatrix}.
\]

The point-on-line constraint is

\[
\Phi_{P,1}
=
\mathbf n_A^T
(\mathbf r_{P_B}-\mathbf r_{P_A})
=0.
\]

If the local axis directions have local orientation angles `α_A` and `α_B`, the orientation constraint is

\[
\Phi_{P,2}
=
\theta_B+\alpha_B-	heta_A-\alpha_A
=0.
\]

This contributes two scalar constraints and leaves one relative translational degree of freedom.

Its natural coordinate is

\[
s_J =
\mathbf a_A^T
(\mathbf r_{P_B}-\mathbf r_{P_A}).
\]

The orientation of `axis_a` determines the sign of `s_J`.

Input axes should be normalized internally; users should not be required to provide unit vectors. Zero-length axes are invalid.

The reference points of a prismatic joint are geometric joint-frame references; they need not correspond to physical pivots or contact points.

## 8. Inputs and drivers

A prescribed input belongs to a **solution problem**, not permanently to the mechanism.

Therefore the same mechanism may be solved with different prescribed joints without mutation.

Preferred high-level form:

```python
solution = solve(
    mechanism,
    input=J1,
    values=theta,
    initial_guess=guess,
)
```

Every MVP joint exposes one natural scalar coordinate `c_J(q)`. A driver simply contributes

\[
\Phi_D(\mathbf q,u)
= c_J(\mathbf q)-u
=0.
\]

No separate public `RotationalDriver` and `TranslationalDriver` classes are required initially.

`0.1.0` only needs one prescribed input in the public API, while the internal formulation should not make future multiple-input support impossible.

Input values parameterize configurations. They do **not** imply physical time.

Therefore an array such as

```python
theta = np.linspace(0, 2*np.pi, 361)
```

means a sequence of prescribed angular coordinates, not necessarily a motion law `θ(t)`.

Animation FPS is a presentation parameter, not a physical angular velocity.

## 9. Coordinate vector and internal indexing

For `n` mobile links,

\[
\mathbf q=
\begin{bmatrix}
\mathbf q_1\\
\mathbf q_2\\
\vdots\\
\mathbf q_n
\end{bmatrix}
\in\mathbb R^{3n}.
\]

A deterministic internal map associates each link with its three entries. The simplest MVP rule is to use link creation order.

The layout is private infrastructure. Public code should query entities instead:

```python
config.pose(coupler)
```

rather than relying on slices of `q`.

## 10. Constraint assembly

For a mechanism with joint constraints

\[
\Phi_J(\mathbf q)=0
\]

and prescribed input

\[
\Phi_D(\mathbf q,u)=0,
\]

the complete position problem is

\[
\boxed{
\Phi(\mathbf q,u)=
\begin{bmatrix}
\Phi_J(\mathbf q)\\
\Phi_D(\mathbf q,u)
\end{bmatrix}
=0
}.
\]

After prescribing the required number of inputs, the MVP expects a square nonlinear system.

The model should be translated into equations by internal mathematical infrastructure. Public joint objects should not require users to manipulate residual vectors or coordinate indices.

## 11. Jacobian

The analytic constraint Jacobian is part of the design from `0.1.0`:

\[
J(\mathbf q,u)
=
\frac{\partial\Phi}{\partial\mathbf q}.
\]

For a local point

\[
\mathbf r = \mathbf p + R(\theta)\mathbf s,
\]

using

\[
E=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix},
\]

we have

\[
\frac{\partial\mathbf r}{\partial\theta}
=R(\theta)E\mathbf s.
\]

This gives simple analytic Jacobian blocks for revolute and prismatic constraints.

The Jacobian is useful not only for nonlinear convergence but also as the foundation for later velocity, acceleration, and singularity analysis.

## 12. Position solver strategy

The initial numerical solver should use a single internal SciPy nonlinear root method, initially based on:

```python
scipy.optimize.root
```

with the analytic Jacobian supplied by Kimech.

No public hierarchy of interchangeable nonlinear solver classes is required in the MVP.

The public API should remain independent enough that the internal method can be changed later without breaking user models.

### 12.1 Solution verification

A SciPy success flag alone is not sufficient.

After convergence, Kimech should recompute

\[
\Phi(\mathbf q^*,u)
\]

and verify the final residual against Kimech's acceptance tolerances before producing a valid `Configuration`.

### 12.2 Scaling

The solver mixes linear coordinates and angles, and the library deliberately does not impose physical units.

An internal characteristic length `L_ref` should therefore be considered to normalize linear coordinates/residuals:

\[
\hat x=x/L_{ref},\qquad
\hat y=y/L_{ref},\qquad
\hat\theta=\theta.
\]

This should remain lightweight internal infrastructure rather than becoming a unit/scaling subsystem.

## 13. Initial guesses and branch selection

A prescribed input does not uniquely identify an assembly branch.

The first solve therefore requires an approximate initial pose for every mobile link.

Preferred user representation:

```python
guess = {
    crank:   (x1, y1, theta1),
    coupler: (x2, y2, theta2),
    rocker:  (x3, y3, theta3),
}
```

Users should not normally construct a raw `3n` vector.

A valid `Configuration` from the same mechanism may also serve as an initial guess.

Partial guesses are not required in `0.1.0`.

The initial guess does not need to satisfy the prescribed coordinate exactly; the driver equation is part of the nonlinear system and corrects it.

Automatic assembly-branch discovery is deliberately deferred.

## 14. Continuation and branch tracking

For a prescribed sequence

\[
u_0,u_1,\ldots,u_N,
\]

the MVP continuation strategy is simple warm starting:

\[
q_0 = \operatorname{solve}(\Phi(q,u_0), q_{guess}),
\]

then

\[
q_k = \operatorname{solve}(\Phi(q,u_k), q_{k-1}).
\]

The order supplied by the user is preserved and defines the traversal direction.

No automatic sorting, adaptive subdivision, predictor-corrector continuation, or branch switching is required for `0.1.0`.

If continuation fails, the default behavior is explicit failure rather than silently inserting NaNs.

## 15. `Configuration`

A `Configuration` represents one accepted mechanism configuration.

Conceptually it contains:

- a reference to the mechanism;
- the solved coordinate vector;
- the associated input value;
- optional solver diagnostics.

It should be conceptually immutable.

Preferred domain queries include:

```python
config.pose(link)
config.position(point)
config.joint_coordinate(joint)
```

Ground should participate uniformly:

```python
config.pose(m.ground)  # identity pose
```

A low-level coordinate array may be exposed for advanced use, but public documentation should favor entity-based queries.

Derived quantities such as every point position should not be redundantly stored by default.

## 16. `KinematicSolution`

A `KinematicSolution` represents an ordered sweep of accepted configurations.

Conceptually:

```text
KinematicSolution
├── mechanism
├── input_joint
├── input_values
├── coordinates Q
└── diagnostics
```

Internally the main state can be stored efficiently as

\[
Q\in\mathbb R^{N\times3n}.
\]

Individual `Configuration` objects may be lightweight views rather than thousands of permanently allocated Python objects.

Candidate queries:

```python
config = solution[i]
path = solution.point_path(P)
poses = solution.link_poses(coupler)
coords = solution.joint_coordinates(J)
```

Point paths should naturally have shape `(N, 2)` and link poses `(N, 3)`.

Trajectories and joint-coordinate histories should be derived from `Q` rather than eagerly duplicated.

Future versions may extend the same solution object with velocity and acceleration information, but `0.1.0` contains position only.

## 17. Failure and diagnostics

A failed nonlinear iterate should not be presented as a valid `Configuration`.

For a sweep, the default behavior is to stop at the first failed configuration and produce a useful diagnostic containing at least:

- failed input index;
- failed input value;
- residual information;
- solver message.

A future exception type may additionally expose the last valid configuration or partial solution.

Silent NaN insertion should not be the default behavior.

Minimal solve diagnostics may include:

- success;
- residual norm/measure;
- iteration or evaluation count;
- underlying solver message.

These diagnostics remain separate conceptually from the kinematic state.

## 18. Mobility and validation

For planar lower-pair mechanisms, Kimech may provide the structural mobility estimate

\[
M=3(N-1)-2J_1-J_2.
\]

For the initial R/P-only scope, both revolute and prismatic pairs are one-DOF lower pairs.

This should be documented as a **structural estimate**, not an absolute guarantee, because redundant constraints and special geometries can invalidate simple counting rules.

A model validation report should eventually check items such as:

- duplicate link names;
- duplicate point names within a body;
- invalid references;
- joints whose two points belong to the same body;
- zero-length prismatic axes;
- disconnected bodies;
- inconsistent driver count versus estimated mobility;
- under- or over-constrained problems where detectable structurally.

The final numerical criterion remains connected to the assembled system and Jacobian behavior.

## 19. Visualization baseline

The core model does not store graphical shapes.

The default `0.1.0` visualization is a **kinematic schematic**, derived from:

- points;
- joint topology;
- joint types;
- solved poses.

The renderer may use conventional glyphs, for example:

- revolute joint → circular pivot marker;
- prismatic joint → slider/guide glyph;
- links → schematic segments connecting structural joint points where possible;
- auxiliary points → optional markers/traces.

The renderer should not pretend to infer the physical CAD shape of a link.

The initial backend should be Matplotlib only. No backend registry or plugin architecture is needed.

Static export should support SVG through Matplotlib. Animation export should initially support GIF, likely through Pillow. MP4/WebM and additional rendering backends are deferred.

## 20. Four-bar acceptance case

A candidate declarative model is:

```python
m = Mechanism("four_bar")

ground = m.ground
A0 = ground.add_point("A", (0.0, 0.0))
D0 = ground.add_point("D", (0.30, 0.0))

crank = m.add_link("crank")
A1 = crank.add_point("A", (0.0, 0.0))
B1 = crank.add_point("B", (0.08, 0.0))

coupler = m.add_link("coupler")
B2 = coupler.add_point("B", (0.0, 0.0))
C2 = coupler.add_point("C", (0.22, 0.0))
P = coupler.add_point("P", (0.10, 0.05))

rocker = m.add_link("rocker")
C3 = rocker.add_point("C", (0.0, 0.0))
D3 = rocker.add_point("D", (0.18, 0.0))

J1 = m.revolute(A0, A1, name="input")
J2 = m.revolute(B1, B2)
J3 = m.revolute(C2, C3)
J4 = m.revolute(D3, D0)
```

There are three mobile links and therefore nine generalized coordinates.

Four revolute joints contribute eight scalar constraints. One prescribed joint coordinate adds the ninth equation.

Acceptance checks should include:

- structural mobility `M = 1`;
- residuals within tolerance;
- constant rigid-link distances;
- coincidence of all revolute joint point pairs;
- continuous branch tracking over a full admissible crank revolution;
- continuous coupler-point path;
- physical equivalence of the first and final configuration after one full revolution, accounting for unwrapped angles;
- no mechanism-specific equations or solver classes.

## 21. Slider-crank acceptance case

A candidate model is:

```python
m = Mechanism("slider_crank")

ground = m.ground
O0 = ground.add_point("O", (0.0, 0.0))

crank = m.add_link("crank")
O1 = crank.add_point("O", (0.0, 0.0))
B1 = crank.add_point("B", (0.08, 0.0))

rod = m.add_link("rod")
B2 = rod.add_point("B", (0.0, 0.0))
C2 = rod.add_point("C", (0.24, 0.0))

slider = m.add_link("slider")
C3 = slider.add_point("C", (0.0, 0.0))

J1 = m.revolute(O0, O1, name="input")
J2 = m.revolute(B1, B2)
J3 = m.revolute(C2, C3)
J4 = m.prismatic(
    O0,
    C3,
    axis_a=(1.0, 0.0),
    axis_b=(1.0, 0.0),
)
```

There are again nine generalized coordinates.

Three revolute joints contribute six constraints and the prismatic joint contributes two. One prescribed coordinate closes the square system.

Acceptance checks should include:

- structural mobility `M = 1`;
- slider reference point remains on the guide;
- slider orientation remains fixed relative to the guide;
- rigid crank and connecting-rod lengths remain constant;
- `J4` returns the expected signed slider displacement;
- full crank revolution can be followed continuously when geometrically admissible;
- an inclined guide works without new entities;
- a rotated slider local frame works by changing `axis_b`, not the solver;
- the prismatically driven inverse case can use the same model;
- no slider-crank-specific solver exists.

## 22. Confirmed design decisions (D1–D54)

### Core model

- **D1.** `Link` is the fundamental mobile rigid body.
- **D2.** Each link owns an arbitrary local frame.
- **D3.** Kinematic geometry is represented through local points rigidly attached to links.
- **D4.** Link lengths are derived quantities, not fundamental properties.
- **D5.** A `Point` belongs to exactly one body and does not store global state.
- **D6.** Point names are unique only within their owning body.
- **D7.** Each mechanism has exactly one ground defining the global frame.
- **D8.** Ground shares the same geometric concept as a rigid link but contributes no generalized coordinates.
- **D9.** Global poses belong to `Configuration`, not `Link`.
- **D10.** Initial guesses belong to the solution problem/solver, not the model.
- **D11.** No premature `RigidBody → Link/Ground` class hierarchy, builder/freeze system, or units subsystem is required.

### Inputs and coordinates

- **D12.** Drivers are not permanent members of `Mechanism`; they belong to a solution problem.
- **D13.** Revolute and prismatic joints expose a natural scalar coordinate conceptually.
- **D14.** A prescribed coordinate contributes `c_J(q) - u = 0`.
- **D15.** Joint type determines angular vs linear semantics; separate rotational/translational driver classes are unnecessary initially.
- **D16.** The high-level candidate API is `solve(mechanism, input=joint, values=...)`.
- **D17.** Public `0.1.0` needs one prescribed input only; the internal design should not preclude multiple inputs later.
- **D18.** Internal angular variables remain unwrapped.
- **D19.** MVP input values parameterize configurations and do not inherently represent time.
- **D20.** Motion laws `u(t)`, `u̇(t)`, and `ü(t)` are deferred.
- **D21.** Input choice does not determine assembly branch; branch selection begins from the initial guess.

### Results

- **D22.** `Configuration` represents one valid solved kinematic configuration.
- **D23.** `Configuration` stores/refs minimal state rather than copying all derived geometry.
- **D24.** `Configuration` is conceptually immutable.
- **D25.** Entity-based queries are preferred over raw coordinate indexing.
- **D26.** `KinematicSolution` represents an ordered sequence of configurations associated with input values.
- **D27.** A dense matrix `Q ∈ R^(N×3n)` is an appropriate internal representation.
- **D28.** Paths, poses, and joint histories are derived rather than eagerly duplicated.
- **D29.** A solution keeps the prescribed input joint and values.
- **D30.** Numerical diagnostics remain conceptually separate from kinematic state.
- **D31.** Sweep failure aborts explicitly by default; silent NaN insertion is not the default.
- **D32.** `KinematicSolution` is independent of the specific nonlinear algorithm that produced it.
- **D33.** The same result model may later grow to contain `q`, `q̇`, and `q̈`; `0.1.0` contains position only.
- **D34.** `input_values` does not imply physical time.
- **D35.** Provisional high-level behavior: scalar `values` may return `Configuration`; a sequence may return `KinematicSolution`.

### Solver

- **D36.** `q` contains three coordinates per mobile link and excludes ground.
- **D37.** The coordinate layout is deterministic private infrastructure; the public API uses model entities.
- **D38.** The complete residual concatenates joint constraints and prescribed-coordinate constraints.
- **D39.** After prescribing required inputs, `0.1.0` expects a square system.
- **D40.** Residual/Jacobian mathematics belongs to the internal mathematical layer, not the public domain model.
- **D41.** Analytic Jacobians are part of the `0.1.0` design.
- **D42.** The initial nonlinear implementation uses one SciPy root method; no multiple-solver abstraction is introduced yet.
- **D43.** Kimech independently verifies final residuals before accepting a configuration.
- **D44.** Lightweight internal length scaling should reduce sensitivity to the user's consistent choice of linear units without introducing a unit system.
- **D45.** Public initial guesses map links to poses rather than exposing raw `q` indexing.
- **D46.** The first solve requires a pose estimate for every mobile link.
- **D47.** A compatible `Configuration` can serve as a new initial guess.
- **D48.** MVP branch tracking uses the previous solved configuration as the next initial guess.
- **D49.** User input order is preserved and determines traversal direction.
- **D50.** Adaptive continuation and automatic branch enumeration are deferred.
- **D51.** A failed sweep stops with explicit diagnostics rather than being hidden.

### Joint/visualization clarifications from acceptance walkthroughs

- **D52.** A `Point` may be referenced by multiple joints.
- **D53.** Prismatic reference points define joint geometry and need not be physical contact/pivot locations.
- **D54.** `0.1.0` schematic visualization is derived from topology, points, joint types, and solved poses; `Link` does not gain graphical shape data.

## 23. Overarchitecture guardrails

Before adding a new abstraction to the MVP, ask:

1. Is it required by four-bar or slider-crank?
2. Does it remove real duplication in the current implementation rather than hypothetical future duplication?
3. Can the same goal be reached with a small private helper instead of a public class?
4. Would removing it make common user code simpler without preventing the known roadmap?

Features that should be resisted until real use cases demand them include:

- backend registries;
- plugin managers;
- generic solver factories;
- symbolic backends;
- automatic model serialization;
- custom unit systems;
- mechanism-specific subclasses such as `FourBar` and `SliderCrank` as core model types;
- mechanism-specific solvers.

Convenience factory functions for common mechanisms may be considered later, but they must build ordinary `Mechanism` objects and use the generic solver.

## 24. Initial roadmap

### `0.1.0` — position kinematics MVP

- mechanism/link/point/ground model;
- revolute and prismatic joints;
- one prescribed joint coordinate;
- structural validation and mobility estimate;
- coordinate map and constraint assembly;
- analytic position Jacobian;
- SciPy root-based position solver;
- explicit full-link initial guess;
- warm-start branch continuation;
- `Configuration`;
- `KinematicSolution`;
- point paths and link/joint queries;
- Matplotlib schematic plotting;
- animation;
- SVG and GIF export;
- four-bar and slider-crank acceptance examples/tests.

### `0.2.0` — differential kinematics

Candidate scope:

- velocity solution from differentiated constraints;
- acceleration solution;
- position/velocity/acceleration histories;
- better Jacobian diagnostics;
- early singularity indicators.

### `0.3.0` — solver robustness

Candidate scope:

- adaptive step subdivision;
- improved initialization workflows;
- better singular/toggle diagnostics;
- predictor-corrector continuation if justified by real failures.

### Later

Only after demonstrated need:

- richer visualization;
- MP4/WebM export;
- additional joint/constraint types;
- second visualization backend;
- common mechanism factory functions;
- synthesis modules consuming the existing model/solver API.

## 25. Open questions intentionally deferred

The following are not blockers for implementation and should remain open until coding or real usage provides evidence:

- exact public parameter names for prismatic axes;
- whether scalar `solve(..., values=x)` returning `Configuration` is preferable to always returning a solution container;
- exact shape/content of diagnostic objects and exceptions;
- algorithm used to estimate a characteristic length;
- convenience helpers for generating initial assembly guesses;
- explicit visual-style objects beyond the default schematic renderer;
- future multiple-driver syntax.

These questions should not delay implementation of the validated core.
