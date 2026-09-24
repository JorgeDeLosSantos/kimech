# Changelog

## 0.5.0

Kimech `0.5.0` consolidates the existing one-DOF planar R/P kinematics core around introspection, solve observability, and analysis.

### Added

- public structural topology snapshots through `Mechanism.topology()` and `MechanismTopology`;
- topology queries for incident joints, adjacent bodies, joints between bodies, connected components, connectivity, degree, and cycle rank;
- schematic topology visualization through `kimech.visualization.plot_topology()`;
- structured numerical failure metadata through `SolveFailureContext` attached optionally to `KinematicSolveError.context`;
- compact solve-history summaries through `SolveDiagnostics.summary()` and `SolveDiagnosticSummary`;
- explicit input-coordinate sensitivity analysis through `input_sensitivity(solution)` and `InputSensitivity`;
- `dq/du`, body-pose, point-position, and natural joint-coordinate sensitivity queries;
- permanent complex-mechanism acceptance coverage for Archimedes trammel, Whitworth quick-return, and Watt II six-bar;
- forward/reverse branch-consistency regression for the Watt II six-bar;
- design/study documents for topology, sensitivity, and regression acceptance.

### Changed

- `KinematicSolution` now retains the joint snapshot associated with the solved mechanism state so later mechanism extensions do not alter downstream sensitivity semantics;
- solve failures at position, velocity, acceleration, and input-tangent stages expose structured context when meaningful while preserving human-readable error messages;
- requested-sample continuation failures report attempted recovery strategies and corrector-attempt counts;
- result diagnostics can be summarized without exposing SciPy-specific counters;
- documentation now distinguishes physical configuration, structural topology, driven-solve diagnostics, and input-coordinate sensitivity explicitly.

### Design limits

- topology visualization is structural/schematic and does not imply physical geometry;
- input sensitivity is computed explicitly after a successful solve and is not a time derivative;
- no universal singularity threshold, binary `is_singular` flag, or dedicated singularity exception is introduced;
- no public Driver abstraction, multiple simultaneous prescribed inputs, general multi-DOF solving, motion-law abstraction, pseudo-arclength continuation, global branch enumeration, or dynamics are included;
- automatic branch-independent initial-guess generation remains deferred;
- Klann and Theo Jansen remain reproducible release-level studies rather than per-commit acceptance tests.


## 0.4.0

Kimech `0.4.0` strengthens continuation robustness and adds structured numerical diagnostics without changing the public `solve()` signature.

### Added

- first-order predictor-corrector continuation using the local input tangent `dq/du`;
- warm-start retry when a predicted nonlinear correction fails;
- bounded adaptive subdivision for recoverable large requested input steps;
- public `SolveDiagnostics` attached to solve-generated `KinematicSolution` objects;
- scaled-Jacobian condition numbers, minimum singular values, and numerical ranks;
- per-sample subdivision counts, accepted solve strategies, nonlinear corrector-attempt counts, and final scaled residual norms;
- singularity/toggle study documenting driver-dependent driven-solve singularities;
- branch-continuity study over Archimedes trammel, Whitworth, Watt II, Klann, and Theo Jansen mechanisms;
- end-to-end acceptance coverage for forward/reverse branch consistency and driver-dependent dead-center diagnostics;
- `docs/design-0.4.0.md` as the solver-robustness design baseline.

### Changed

- position sweeps now prefer tangent prediction before the nonlinear corrector while preserving warm-start behavior as a fallback;
- failed requested steps may be recovered internally through hidden midpoint solves without changing the returned requested-sample history;
- solve diagnostics are computed from the dimensionless scaled driven Jacobian rather than the raw dimensional Jacobian;
- exact rank loss, conditioning, and minimum singular value are treated as descriptive properties of the selected driven formulation rather than as a universal physical-singularity classification;
- diagnostics are preserved when slicing a `KinematicSolution`.

### Design limits

- adaptive subdivision does not make unreachable targets solvable and cannot cross folds where the selected input coordinate ceases to parameterize the branch;
- no universal near-singularity threshold or public `is_singular` flag is defined;
- no global assembly-mode identifier or automatic branch enumeration is introduced;
- pseudo-arclength continuation remains deferred;
- SciPy-specific counters such as `nfev` and `njev` are not part of the public diagnostics API;
- multiple prescribed inputs, automatic initial-guess generation, dynamics, forces, masses, and inertias remain out of scope.


## 0.3.0

Kimech `0.3.0` consolidates numerical robustness and simplifies the public solve API.

### Changed

- `solve()` now uses `input_joint` and `input_position` instead of `input` and `values`;
- `solve()` always returns `KinematicSolution`, including scalar prescribed positions;
- scalar prescribed positions produce a one-sample solution and `solution[0]` returns the corresponding `Configuration`;
- `Configuration.input_value` was renamed to `Configuration.input_position`;
- `KinematicSolution.input_values` was renamed to `KinematicSolution.input_positions`;
- position, velocity, and acceleration solves use internal dimensionless numerical scaling and scale-independent residual validation;
- point queries are named consistently as `point_position(s)`, `point_velocity/velocities`, and `point_acceleration/accelerations`;
- `KinematicSolution` supports slicing to another `KinematicSolution` and explicit iteration over `Configuration` objects;
- result containers remain externally immutable and may be empty when constructed manually;
- prescribed-input metadata on `Configuration` requires an associated `input_joint`;
- visualization uses explicit body render scaffolds with fallback support for plate-like links and lightweight connectors for auxiliary points;
- joint glyph scaling is derived from body scaffolds rather than arbitrary remote auxiliary points;
- `animate()` supports opt-in progressive point traces through `trace_points=[...]`.

These are intentional breaking changes. No compatibility aliases are provided because Kimech has not yet established an external user base or PyPI release workflow.

### Still out of scope

- multiple prescribed inputs or a public Driver abstraction;
- automatic initial-guess generation or branch discovery;
- formal singularity diagnostics;
- dynamics, forces, torques, masses, or inertias.


## 0.2.0

Kimech `0.2.0` extends the initial position-kinematics MVP with differential kinematics for the same one-DOF planar rigid-body mechanisms with revolute and prismatic joints.

### Added

- prescribed input velocity and acceleration through `solve()`;
- generalized coordinate velocity and acceleration state;
- body velocity and acceleration queries;
- point velocity and acceleration queries;
- joint velocity and acceleration queries;
- velocity and acceleration histories in `KinematicSolution`;
- analytic differential kinematics using the existing constraint Jacobian;
- independent residual verification for velocity and acceleration linear solves;
- acceptance coverage for four-bar and slider-crank mechanisms, including prismatically driven inverse kinematics.

### Changed

- `Configuration.pose()` was renamed to `Configuration.body_pose()`;
- `KinematicSolution.link_poses()` was renamed to `KinematicSolution.body_poses()`;
- result objects can now retain optional `q_dot` / `q_ddot` state and prescribed differential input metadata.

The two method renames are intentional breaking changes. No compatibility aliases are provided because Kimech is still pre-`1.0`.

### Still out of scope

- multiple prescribed inputs;
- formal singularity diagnostics;
- adaptive continuation or automatic branch discovery;
- dynamics, forces, torques, masses, or inertias;
- explicit time arrays or motion-law objects;
- additional visualization backends.

## 0.1.0

Initial position-kinematics MVP for one-DOF planar rigid-body mechanisms with revolute and prismatic joints, including warm-start continuation, result queries, plotting, and animation.
